"""Safety-oriented high-level control for a motorized rotator."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from .protocol import (
    ConfigError,
    PA,
    P3,
    P4,
    SafetyError,
    RotatorError,
    STATUS_COUNT,
    STATUS_START,
    VIRTUAL_IO_PROFILE,
    VIRTUAL_IO_MIXED_PROFILE,
    ControlMode,
    PositionCommandMode,
    PositionInputMode,
    SpeedSource,
    TorqueSource,
    VirtualInput,
    decode_i16,
    decode_i32,
    describe_alarm,
    encode_i16,
    p3,
    p4,
    pa,
    slot_registers,
)


class RegisterClient(Protocol):
    def read(self, address: int, count: int = 1) -> list[int]: ...

    def write(self, address: int, value: int) -> None: ...


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    speed: int = 60
    torque: int = 10
    duration: float = 10.0
    ramp: int = 500
    margin: int = 0
    error: int = 1000
    threshold: int = 2
    timeout: float = 5.0

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if not 1 <= self.torque <= 300:
            raise ValueError("torque must be 1..300")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.ramp <= 0:
            raise ValueError("ramp must be positive")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.error <= 0:
            raise ValueError("error must be positive")
        if self.threshold < 0 or self.timeout <= 0:
            raise ValueError("stop thresholds must be non-negative/positive")


@dataclass(frozen=True, slots=True)
class RotatorStatus:
    motor_speed_rpm: int
    position_pulses: int
    position_command_pulses: int
    position_error_pulses: int
    motor_torque_percent: int
    motor_current_raw: int
    control_mode: int
    temperature_raw: int
    speed_command_rpm: int
    torque_command_percent: int
    rotor_absolute_position: int
    input_state: int
    output_state: int
    encoder_input_state: int
    bus_voltage_v: int
    alarm_code: int
    logic_version: int
    relay_state: int
    run_state: int
    external_voltage_state: int
    absolute_position: int

    def data(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckResult:
    device_id: int
    baudrate: int
    serial_format: int
    rs485_enabled: bool
    force_enable_active: bool
    status: RotatorStatus

    def data(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.data()
        return result


@dataclass(frozen=True, slots=True)
class MotionResult:
    requested_value: int
    duration_seconds: float
    samples: tuple[RotatorStatus, ...]
    final_status: RotatorStatus
    elapsed_seconds: float | None = None

    def data(self) -> dict[str, object]:
        return {
            "requested_value": self.requested_value,
            "duration_seconds": self.duration_seconds,
            "samples": [sample.data() for sample in self.samples],
            "final_status": self.final_status.data(),
            "elapsed_seconds": self.elapsed_seconds,
        }

    def summary(self) -> dict[str, object]:
        return {
            "requested_value": self.requested_value,
            "requested_duration_seconds": self.duration_seconds,
            "motion_seconds": self.elapsed_seconds,
            "sample_count": len(self.samples),
            "peak_speed_rpm": max(
                (abs(sample.motor_speed_rpm) for sample in self.samples),
                default=0,
            ),
            "peak_torque_percent": max(
                (abs(sample.motor_torque_percent) for sample in self.samples),
                default=0,
            ),
            "final_status": self.final_status.data(),
        }


@dataclass(frozen=True, slots=True)
class Calculation:
    angle: float
    speed: float
    ratio: float
    estimate: float
    revolutions: float
    commands: int
    turns: int
    pulses: int
    rpm: int
    ramp: int

    def data(self) -> dict[str, int | float]:
        return asdict(self)


def calculate(
    *,
    angle: float,
    speed: float,
    ratio: float,
    resolution: int,
    ramp: int,
) -> Calculation:
    """Convert rotator angle and angular speed to motor-side position values."""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    if not math.isfinite(speed) or not 0 < speed <= 30:
        raise ValueError("speed must be >0 and <=30 degrees/second")
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be finite and positive")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if not 1 <= ramp <= 10000:
        raise ValueError("ramp must be 1..10000")

    # 传动比表示输出轴每转一圈对应的电机转数
    # 比例为一百八比一时转台转九十度需要电机转四十五圈
    revolutions = angle * ratio / 360.0
    commands = round(revolutions * resolution)
    turns = math.trunc(commands / resolution)
    pulses = commands - turns * resolution
    rpm = math.ceil(speed * ratio / 6.0)
    estimate = abs(angle) / speed + ramp * rpm / 1_000_000.0
    return Calculation(
        angle=angle,
        speed=speed,
        ratio=ratio,
        estimate=estimate,
        revolutions=revolutions,
        commands=commands,
        turns=turns,
        pulses=pulses,
        rpm=rpm,
        ramp=ramp,
    )


@dataclass(frozen=True, slots=True)
class MoveResult:
    calculation: Calculation
    motion_seconds: float | None
    operation_seconds: float
    peak_speed_rpm: int
    peak_torque_percent: int
    final_status: RotatorStatus

    def data(self) -> dict[str, object]:
        return {
            "calculation": self.calculation.data(),
            "motion_seconds": self.motion_seconds,
            "operation_seconds": self.operation_seconds,
            "peak_speed_rpm": self.peak_speed_rpm,
            "peak_torque_percent": self.peak_torque_percent,
            "final_status": self.final_status.data(),
        }


class _Controller:
    """High-level commands with explicit motion and persistence interlocks."""

    def __init__(
        self,
        client: RegisterClient,
        *,
        device: int = 1,
        baudrate: int = 9600,
        format: int = 0,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.client = client
        self.device = device
        self.baudrate = baudrate
        self.format = format
        self.limits = limits or SafetyLimits()
        self._restart_prepared = False

    def _read_pa(self, number: int | PA, *, signed: bool = False) -> int:
        value = self._read_one(pa(int(number)))
        return decode_i16(value) if signed else value

    def _read_p3(self, number: int | P3, *, signed: bool = False) -> int:
        value = self._read_one(p3(int(number)))
        return decode_i16(value) if signed else value

    def _read_p4(self, number: int | P4, *, signed: bool = False) -> int:
        value = self._read_one(p4(int(number)))
        return decode_i16(value) if signed else value

    def status(self) -> RotatorStatus:
        values = self.client.read(STATUS_START, STATUS_COUNT)
        if len(values) != STATUS_COUNT:
            raise RotatorError(
                f"status read returned {len(values)} registers, expected {STATUS_COUNT}"
            )
        return RotatorStatus(
            motor_speed_rpm=decode_i16(values[0]),
            position_pulses=decode_i32([values[2], values[1]]),
            position_command_pulses=decode_i32([values[4], values[3]]),
            position_error_pulses=decode_i32([values[6], values[5]]),
            motor_torque_percent=decode_i16(values[7]),
            motor_current_raw=decode_i16(values[8]),
            control_mode=values[9],
            temperature_raw=decode_i16(values[10]),
            speed_command_rpm=decode_i16(values[11]),
            torque_command_percent=decode_i16(values[12]),
            rotor_absolute_position=decode_i32([values[14], values[13]]),
            input_state=values[15],
            output_state=values[16],
            encoder_input_state=values[17],
            bus_voltage_v=values[18],
            alarm_code=values[19],
            logic_version=values[20],
            relay_state=values[21],
            run_state=values[22],
            external_voltage_state=values[23],
            absolute_position=self._decode_i64(values[24:28]),
        )

    def check(self, *, stopped: bool = True, alarm: bool = False) -> CheckResult:
        report = CheckResult(
            device_id=self._read_pa(PA.MODBUS_ADDRESS),
            baudrate=self._read_pa(PA.MODBUS_BAUDRATE) * 100,
            serial_format=self._read_pa(PA.MODBUS_FORMAT),
            rs485_enabled=self._read_pa(PA.RS485_ENABLE) == 0,
            force_enable_active=self._read_pa(PA.FORCE_ENABLE) == 1,
            status=self.status(),
        )
        failures = []
        if report.device_id != self.device:
            failures.append(
                f"drive address is {report.device_id}, client expects {self.device}"
            )
        if report.baudrate != self.baudrate:
            failures.append(
                f"drive baudrate is {report.baudrate}, client expects {self.baudrate}"
            )
        if report.serial_format != self.format:
            failures.append(
                f"drive serial format is {report.serial_format}, expected {self.format}"
            )
        if not report.rs485_enabled:
            failures.append("PA104 disables RS485")
        if report.status.alarm_code and not alarm:
            code = report.status.alarm_code
            failures.append(f"drive alarm is {code} ({describe_alarm(code)})")
        if stopped and abs(report.status.motor_speed_rpm) > self.limits.threshold:
            failures.append(f"motor is already moving at {report.status.motor_speed_rpm} rpm")
        if failures:
            raise SafetyError("safety check failed: " + "; ".join(failures))
        return report

    def disable(self) -> RotatorStatus:
        """Remove software forced enable without changing EEPROM."""
        self.stop()
        self._write_pa(PA.FORCE_ENABLE, 0, temporary=True)
        self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
        return self._wait_stopped()

    def stop(self) -> RotatorStatus:
        """Best-effort communication stop; hardware E-stop is still required."""
        errors = []
        try:
            self._write_p3(
                P3.VIRTUAL_INPUT_STATE,
                int(VirtualInput.POSITION_HOLD),
            )
        except Exception as exc:
            errors.append(exc)
        for parameter in (PA.INTERNAL_SPEED_1, PA.INTERNAL_TORQUE_1):
            try:
                self._write_pa(parameter, 0, temporary=True)
            except Exception as exc:  # 单次写入失败后继续处理
                errors.append(exc)
        if errors:
            raise RotatorError(
                "communication stop was incomplete; use the hardware power cut/E-stop"
            ) from errors[-1]
        return self._wait_stopped()

    def clear(self, *, confirm: str) -> RotatorStatus:
        """Clear a resettable alarm only after zeroing commands and disabling."""
        self._require_confirmation(confirm, "CLEAR")
        status = self.status()
        if abs(status.motor_speed_rpm) > self.limits.threshold:
            raise SafetyError("cannot clear an alarm while the motor is moving")
        self.disable()
        self.client.write(pa(PA.ALARM_CLEAR, temporary=True), 1)
        time.sleep(0.15)
        self.client.write(pa(PA.ALARM_CLEAR, temporary=True), 0)
        time.sleep(0.15)
        status = self.status()
        if status.alarm_code:
            code = status.alarm_code
            raise RotatorError(
                f"alarm {code} ({describe_alarm(code)}) did not clear; "
                "power-cycle or inspect the drive"
            )
        return status

    def _enable(self, *, confirm: str) -> RotatorStatus:
        self._require_confirmation(confirm, "ENABLE")
        status = self.check(stopped=True).status
        self._write_pa(PA.FORCE_ENABLE, 1, temporary=True)
        time.sleep(0.15)
        enabled = self.status()
        if enabled.alarm_code:
            self._write_pa(PA.FORCE_ENABLE, 0, temporary=True)
            code = enabled.alarm_code
            raise RotatorError(
                f"drive alarmed while enabling: {code} ({describe_alarm(code)})"
            )
        return enabled

    def mode(
        self,
        mode: ControlMode,
        *,
        speed: int = 60,
        torque: int = 10,
        ramp: int = 1000,
        confirm: str,
    ) -> None:
        """Persist a disabled, zero-command profile; restart separately to apply PA4."""
        self._require_confirmation(confirm, "PERSIST")
        try:
            mode = ControlMode(mode)
        except ValueError as exc:
            raise ValueError("control mode must be PA4 value 0..5") from exc
        self._validate_speed(speed, allow_zero=False)
        if speed < 1:
            raise SafetyError("maximum speed must be positive")
        self._validate_limit(torque)
        if not self.limits.ramp <= ramp <= 10000:
            raise SafetyError(
                f"ramp must be {self.limits.ramp}..10000 ms"
            )
        self.check(stopped=True, alarm=True)

        # PA4 只有重启后才会生效
        # 先保存零指令并禁用驱动器避免下次启动时运动
        self._write_pa(PA.FORCE_ENABLE, 0, temporary=False)
        self._write_pa(PA.INTERNAL_SPEED_1, 0, temporary=False)
        self._write_pa(PA.INTERNAL_TORQUE_1, 0, temporary=False)
        self._write_pa(PA.MAXIMUM_SPEED, speed, temporary=False)
        self._write_pa(
            PA.INTERNAL_CCW_TORQUE_LIMIT, torque, temporary=False
        )
        self._write_pa(
            PA.INTERNAL_CW_TORQUE_LIMIT, -torque, temporary=False
        )
        self._write_pa(PA.ACCELERATION_TIME, ramp, temporary=False)
        self._write_pa(PA.DECELERATION_TIME, ramp, temporary=False)
        self._write_pa(PA.S_CURVE_TIME, min(100, ramp), temporary=False)
        self._write_pa(PA.SPEED_SOURCE, SpeedSource.INTERNAL, temporary=False)
        self._write_pa(PA.TORQUE_SOURCE, TorqueSource.INTERNAL, temporary=False)
        self._write_pa(PA.COMMUNICATION_ERROR_ACTION, 1, temporary=False)
        self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
        if mode in {
            ControlMode.POSITION,
            ControlMode.POSITION_SPEED,
            ControlMode.POSITION_TORQUE,
        }:
            self._write_pa(
                PA.POSITION_INPUT_MODE, PositionInputMode.INTERNAL, temporary=False
            )
        self._write_pa(PA.CONTROL_MODE, mode, temporary=False)
        self._restart_prepared = True

    def restart(self, *, confirm: str) -> RotatorStatus:
        """Restart the drive after a safely prepared persistent mode change."""
        self._require_confirmation(confirm, "RESTART")
        if not self._restart_prepared:
            raise SafetyError(
                "restart is only allowed in the same controller session immediately "
                "after mode"
            )
        self.disable()
        if self._read_pa(PA.FORCE_ENABLE) != 0:
            raise SafetyError("refusing restart because PA53 is not zero")
        if self._read_pa(PA.INTERNAL_SPEED_1, signed=True) != 0:
            raise SafetyError("refusing restart because PA24 is not zero")
        if self._read_pa(PA.INTERNAL_TORQUE_1, signed=True) != 0:
            raise SafetyError("refusing restart because PA64 is not zero")

        target_mode = ControlMode(self._read_pa(PA.CONTROL_MODE))
        primary_modes = {
            ControlMode.POSITION: ControlMode.POSITION,
            ControlMode.SPEED: ControlMode.SPEED,
            ControlMode.TORQUE: ControlMode.TORQUE,
            ControlMode.POSITION_SPEED: ControlMode.POSITION,
            ControlMode.POSITION_TORQUE: ControlMode.POSITION,
            ControlMode.SPEED_TORQUE: ControlMode.SPEED,
        }
        self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
        self._restart_prepared = False
        self.client.write(pa(PA.SOFT_RESET), 1)
        deadline = time.monotonic() + 8.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            time.sleep(0.25)
            try:
                status = self.status()
            except Exception as exc:  # 重启期间驱动器会暂时离线
                last_error = exc
                continue
            if status.alarm_code:
                code = status.alarm_code
                raise RotatorError(
                    f"drive restarted with alarm {code} ({describe_alarm(code)})"
                )
            if abs(status.motor_speed_rpm) > self.limits.threshold:
                raise SafetyError(
                    "drive moved during restart; use the hardware power cut/E-stop"
                )
            if status.control_mode != int(primary_modes[target_mode]):
                continue
            if self._read_pa(PA.FORCE_ENABLE) != 0:
                raise SafetyError("drive restarted with PA53 force enable active")
            return status
        raise RotatorError("drive did not return after software restart") from last_error

    def _configure_speed(
        self,
        *,
        maximum_rpm: int,
        torque_limit_percent: int,
        ramp_ms: int,
    ) -> None:
        self._validate_speed(maximum_rpm, allow_zero=False)
        if maximum_rpm < 1:
            raise SafetyError("maximum speed must be positive")
        self._validate_limit(torque_limit_percent)
        if not self.limits.ramp <= ramp_ms <= 10000:
            raise SafetyError(
                f"ramp must be {self.limits.ramp}..10000 ms"
            )
        self._write_pa(PA.INTERNAL_SPEED_1, 0, temporary=True)
        self._write_pa(PA.MAXIMUM_SPEED, maximum_rpm, temporary=True)
        self._write_pa(
            PA.INTERNAL_CCW_TORQUE_LIMIT, torque_limit_percent, temporary=True
        )
        self._write_pa(
            PA.INTERNAL_CW_TORQUE_LIMIT, -torque_limit_percent, temporary=True
        )
        self._write_pa(PA.ACCELERATION_TIME, ramp_ms, temporary=True)
        self._write_pa(PA.DECELERATION_TIME, ramp_ms, temporary=True)
        self._write_pa(PA.S_CURVE_TIME, min(100, ramp_ms), temporary=True)
        self._write_pa(PA.SPEED_SOURCE, SpeedSource.INTERNAL, temporary=True)

    def _set_speed(self, rpm: int, *, confirm: str | None = None) -> None:
        self._validate_speed(rpm, allow_zero=True)
        if rpm:
            self._require_confirmation(confirm, "MOVE")
        self._write_pa(PA.INTERNAL_SPEED_1, rpm, temporary=True)

    def speed(
        self,
        rpm: int,
        duration: float,
        *,
        limit: int | None = None,
        torque: int | None = None,
        ramp: int = 1000,
        confirm: str,
    ) -> MotionResult:
        self._require_confirmation(confirm, "MOVE")
        self._validate_duration(duration)
        self._validate_speed(rpm, allow_zero=False)
        limit = self.limits.speed if limit is None else limit
        torque = torque or self.limits.torque
        if limit < 1:
            raise SafetyError("maximum speed must be positive")
        self._validate_overspeed(abs(rpm), limit)

        report = self.check(stopped=True)
        self._require_mode(report.status, ControlMode.SPEED)
        restore = self._snapshot_pa(
            PA.SPEED_SOURCE,
            PA.MAXIMUM_SPEED,
            PA.INTERNAL_SPEED_1,
            PA.INTERNAL_CCW_TORQUE_LIMIT,
            PA.INTERNAL_CW_TORQUE_LIMIT,
            PA.ACCELERATION_TIME,
            PA.DECELERATION_TIME,
            PA.S_CURVE_TIME,
        )
        samples = []
        try:
            self._configure_speed(
                maximum_rpm=limit,
                torque_limit_percent=torque,
                ramp_ms=ramp,
            )
            self._enable(confirm="ENABLE")
            self._set_speed(rpm, confirm="MOVE")
            started = time.monotonic()
            while time.monotonic() - started < duration:
                status = self.status()
                samples.append(status)
                if status.alarm_code:
                    code = status.alarm_code
                    raise RotatorError(
                        f"drive alarm during speed motion: "
                        f"{code} ({describe_alarm(code)})"
                    )
                time.sleep(min(0.1, duration / 4))
            motion_elapsed = time.monotonic() - started
        finally:
            self._write_pa(PA.INTERNAL_SPEED_1, 0, temporary=True)
            try:
                self._wait_stopped()
            finally:
                self._restore_pa(restore)
        final_status = self.status()
        return MotionResult(
            rpm,
            duration,
            tuple(samples),
            final_status,
            motion_elapsed,
        )

    def _configure_torque(
        self, *, speed_limit_rpm: int, torque_limit_percent: int
    ) -> None:
        self._validate_speed(speed_limit_rpm, allow_zero=False)
        if speed_limit_rpm < 1:
            raise SafetyError("torque-mode speed limit must be positive")
        self._validate_limit(torque_limit_percent)
        self._write_pa(PA.INTERNAL_TORQUE_1, 0, temporary=True)
        self._write_pa(PA.TORQUE_MODE_SPEED_LIMIT, speed_limit_rpm, temporary=True)
        self._write_pa(
            PA.INTERNAL_CCW_TORQUE_LIMIT, torque_limit_percent, temporary=True
        )
        self._write_pa(
            PA.INTERNAL_CW_TORQUE_LIMIT, -torque_limit_percent, temporary=True
        )
        self._write_pa(PA.TORQUE_SOURCE, TorqueSource.INTERNAL, temporary=True)

    def _set_torque(self, percent: int, *, confirm: str | None = None) -> None:
        self._validate_torque(percent)
        if percent:
            self._require_confirmation(confirm, "TORQUE")
        self._write_pa(PA.INTERNAL_TORQUE_1, percent, temporary=True)

    def torque(
        self,
        percent: int,
        duration: float,
        *,
        speed: int,
        confirm: str,
    ) -> MotionResult:
        self._require_confirmation(confirm, "TORQUE")
        self._validate_duration(duration)
        self._validate_torque(percent)
        if percent == 0:
            raise SafetyError("torque command must not be zero")
        report = self.check(stopped=True)
        self._require_mode(report.status, ControlMode.TORQUE)
        restore = self._snapshot_pa(
            PA.TORQUE_SOURCE,
            PA.TORQUE_MODE_SPEED_LIMIT,
            PA.INTERNAL_TORQUE_1,
            PA.INTERNAL_CCW_TORQUE_LIMIT,
            PA.INTERNAL_CW_TORQUE_LIMIT,
        )
        samples = []
        try:
            self._configure_torque(
                speed_limit_rpm=speed,
                torque_limit_percent=max(abs(percent), 1),
            )
            self._enable(confirm="ENABLE")
            self._set_torque(percent, confirm="TORQUE")
            started = time.monotonic()
            while time.monotonic() - started < duration:
                status = self.status()
                samples.append(status)
                if status.alarm_code:
                    code = status.alarm_code
                    raise RotatorError(
                        f"drive alarm during torque motion: "
                        f"{code} ({describe_alarm(code)})"
                    )
                time.sleep(min(0.1, duration / 4))
            motion_elapsed = time.monotonic() - started
        finally:
            self._write_pa(PA.INTERNAL_TORQUE_1, 0, temporary=True)
            self._write_pa(PA.FORCE_ENABLE, 0, temporary=True)
            self._restore_pa(restore)
        return MotionResult(
            percent,
            duration,
            tuple(samples),
            self.status(),
            motion_elapsed,
        )

    def io(self, *, profile: str = "motion", confirm: str) -> None:
        """Persist a profile while retaining the four physical DI inputs."""
        self._require_confirmation(confirm, "PERSIST")
        profiles = {
            "motion": VIRTUAL_IO_PROFILE,
            "mixed": VIRTUAL_IO_MIXED_PROFILE,
        }
        if profile not in profiles:
            raise ValueError("virtual IO profile must be 'motion' or 'mixed'")
        if abs(self.status().motor_speed_rpm) > self.limits.threshold:
            raise SafetyError("cannot configure virtual IO while the motor is moving")
        self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
        for offset, function in enumerate(profiles[profile]):
            self._write_p3(int(P3.VIRTUAL_DI1_FUNCTION) + offset, int(function))
        self._write_p3(P3.VIRTUAL_IO_MODE, 2)

    def _io_ready(self, profile: str | None = None) -> bool:
        if self._read_p3(P3.VIRTUAL_IO_MODE) != 2:
            return False
        actual = self.client.read(
            p3(P3.VIRTUAL_DI1_FUNCTION), len(VIRTUAL_IO_PROFILE)
        )
        profiles = {
            "motion": VIRTUAL_IO_PROFILE,
            "mixed": VIRTUAL_IO_MIXED_PROFILE,
        }
        if profile is not None:
            if profile not in profiles:
                raise ValueError("virtual IO profile must be 'motion' or 'mixed'")
            candidates = (profiles[profile],)
        else:
            candidates = tuple(profiles.values())
        return any(actual == [int(function) for function in item] for item in candidates)

    def mixed(
        self,
        mode: ControlMode,
        *,
        secondary: bool = False,
        confirm: str,
    ) -> RotatorStatus:
        """Select PA4=3/4/5 and set the virtual CMODE input while stopped."""
        self._require_confirmation(confirm, "MODE")
        try:
            mode = ControlMode(mode)
        except ValueError as exc:
            raise ValueError("mixed mode must be POSITION_SPEED, POSITION_TORQUE, or SPEED_TORQUE") from exc
        if mode not in {
            ControlMode.POSITION_SPEED,
            ControlMode.POSITION_TORQUE,
            ControlMode.SPEED_TORQUE,
        }:
            raise ValueError("only PA4 mixed control modes 3, 4, and 5 are accepted")
        if not self._io_ready("mixed"):
            raise ConfigError(
                "mixed virtual IO profile is not configured"
            )
        self.check(stopped=True)
        if self._read_pa(PA.CONTROL_MODE) != int(mode):
            raise ConfigError(
                f"saved PA4 is not {int(mode)}"
            )
        state = int(VirtualInput.MODE_SELECT) if secondary else 0
        self._write_p3(P3.VIRTUAL_INPUT_STATE, state)
        time.sleep(0.05)
        status = self.status()
        active_modes = {
            ControlMode.POSITION_SPEED: (ControlMode.POSITION, ControlMode.SPEED),
            ControlMode.POSITION_TORQUE: (ControlMode.POSITION, ControlMode.TORQUE),
            ControlMode.SPEED_TORQUE: (ControlMode.SPEED, ControlMode.TORQUE),
        }
        expected = active_modes[mode][int(secondary)]
        self._require_mode(status, expected)
        return status

    def jog(
        self,
        direction: str,
        rpm: int,
        duration: float,
        *,
        confirm: str,
    ) -> MotionResult:
        self._require_confirmation(confirm, "MOVE")
        self._require_io()
        self._validate_duration(duration)
        self._validate_speed(rpm, allow_zero=False)
        if rpm < 1:
            raise SafetyError("JOG speed must be positive; direction is a separate argument")
        self._validate_overspeed(rpm, self.limits.speed)
        report = self.check(stopped=True)
        self._require_mode(report.status, ControlMode.SPEED)
        if direction not in {"positive", "negative"}:
            raise ValueError("direction must be positive or negative")
        bit = (
            VirtualInput.JOG_POSITIVE
            if direction == "positive"
            else VirtualInput.JOG_NEGATIVE
        )
        restore = self._snapshot_pa(
            PA.SPEED_SOURCE, PA.JOG_SPEED, PA.MAXIMUM_SPEED
        )
        samples = []
        try:
            self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
            self._write_pa(PA.JOG_SPEED, abs(rpm), temporary=True)
            self._write_pa(
                PA.MAXIMUM_SPEED, self.limits.speed, temporary=True
            )
            self._write_pa(PA.SPEED_SOURCE, SpeedSource.IO_JOG, temporary=True)
            self._enable(confirm="ENABLE")
            self._write_p3(P3.VIRTUAL_INPUT_STATE, int(bit))
            started = time.monotonic()
            while time.monotonic() - started < duration:
                status = self.status()
                samples.append(status)
                if status.alarm_code:
                    code = status.alarm_code
                    raise RotatorError(
                        f"drive alarm during JOG motion: "
                        f"{code} ({describe_alarm(code)})"
                    )
                time.sleep(min(0.1, duration / 4))
            motion_elapsed = time.monotonic() - started
        finally:
            self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
            try:
                self._wait_stopped()
            finally:
                self._restore_pa(restore)
        return MotionResult(
            rpm,
            duration,
            tuple(samples),
            self.status(),
            motion_elapsed,
        )

    def slot(
        self,
        slot: int,
        *,
        turns: int,
        pulses: int,
        rpm: int,
        mode: PositionCommandMode = PositionCommandMode.INCREMENTAL,
        confirm: str,
    ) -> None:
        self._require_confirmation(confirm, "PERSIST")
        self._validate_speed(rpm, allow_zero=False)
        if rpm < 1:
            raise SafetyError("position speed must be positive")
        if not -30000 <= turns <= 30000:
            raise ValueError("turns must be between -30000 and 30000")
        if not -32768 <= pulses <= 32767:
            raise ValueError(
                "pulses must fit signed 16-bit; also respect the configured pulses/revolution"
            )
        turns_address, pulses_address, speed_address = slot_registers(slot)
        self._write_verified(p4(P4.POSITION_COMMAND_MODE), int(mode))
        self._write_verified(turns_address, self._word(turns))
        self._write_verified(pulses_address, self._word(pulses))
        self._write_verified(speed_address, rpm)

    def run(
        self,
        slot: int,
        *,
        timeout: float,
        torque: int | None = None,
        confirm: str,
    ) -> MotionResult:
        self._require_confirmation(confirm, "MOVE")
        self._require_io()
        self._validate_duration(timeout)
        torque = (
            self.limits.torque
            if torque is None
            else torque
        )
        self._validate_limit(torque)
        _, _, speed_address = slot_registers(slot)
        report = self.check(stopped=True)
        self._require_mode(report.status, ControlMode.POSITION)
        self._require_position(report.status)
        slot_speed = self._read_one(speed_address)
        self._validate_overspeed(
            slot_speed, self._read_pa(PA.MAXIMUM_SPEED)
        )
        restore = self._snapshot_pa(
            PA.POSITION_INPUT_MODE,
            PA.INTERNAL_CCW_TORQUE_LIMIT,
            PA.INTERNAL_CW_TORQUE_LIMIT,
        )
        selection = slot - 1
        state = 0
        if selection & 1:
            state |= int(VirtualInput.POSITION_SELECT_0)
        if selection & 2:
            state |= int(VirtualInput.POSITION_SELECT_1)
        if selection & 4:
            state |= int(VirtualInput.POSITION_SELECT_2)
        samples = []
        start_status = self.status()
        try:
            self._write_p3(P3.VIRTUAL_INPUT_STATE, state)
            self._write_pa(PA.POSITION_INPUT_MODE, PositionInputMode.INTERNAL, temporary=True)
            self._write_pa(
                PA.INTERNAL_CCW_TORQUE_LIMIT,
                torque,
                temporary=True,
            )
            self._write_pa(
                PA.INTERNAL_CW_TORQUE_LIMIT,
                -torque,
                temporary=True,
            )
            self._enable(confirm="ENABLE")
            self._write_p3(
                P3.VIRTUAL_INPUT_STATE, state | int(VirtualInput.POSITION_TRIGGER)
            )
            self._write_p3(P3.VIRTUAL_INPUT_STATE, state)
            started = time.monotonic()
            motion_observed = False
            stopped_samples = 0
            while time.monotonic() - started < timeout:
                status = self.status()
                samples.append(status)
                if status.alarm_code:
                    code = status.alarm_code
                    raise RotatorError(
                        f"drive alarm during position motion: "
                        f"{code} ({describe_alarm(code)})"
                    )
                motion_observed = motion_observed or any(
                    (
                        abs(status.motor_speed_rpm) > self.limits.threshold,
                        status.position_pulses != start_status.position_pulses,
                        status.position_command_pulses
                        != start_status.position_command_pulses,
                    )
                )
                if (
                    motion_observed
                    and abs(status.motor_speed_rpm) <= self.limits.threshold
                    and abs(status.position_error_pulses) <= 130
                ):
                    stopped_samples += 1
                    if stopped_samples >= 3:
                        break
                else:
                    stopped_samples = 0
                time.sleep(0.1)
            else:
                raise SafetyError(
                    "position motion timed out; restart while disabled before retrying"
                )
            motion_elapsed = time.monotonic() - started
        finally:
            self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
            try:
                self._wait_stopped()
            finally:
                self._restore_pa(restore)
        return MotionResult(
            slot,
            timeout,
            tuple(samples),
            self.status(),
            motion_elapsed,
        )

    def homing(
        self,
        *,
        source: int,
        final: int,
        high: int,
        low: int,
        stop: int = 0,
        turns: int = 0,
        pulses: int = 0,
        confirm: str,
    ) -> None:
        self._require_confirmation(confirm, "PERSIST")
        if not 0 <= source <= 5:
            raise ValueError("source must be 0..5")
        if not 0 <= final <= 2:
            raise ValueError("final must be 0..2")
        if stop not in (0, 1):
            raise ValueError("stop must be 0 or 1")
        self._validate_speed(high, allow_zero=False)
        if high < 1:
            raise SafetyError("high homing speed must be positive")
        if not 1 <= low <= min(500, high):
            raise SafetyError("low homing speed must be 1..min(500, high speed)")
        if not -30000 <= turns <= 30000:
            raise ValueError("turns must be -30000..30000")
        if not -32768 <= pulses <= 32767:
            raise ValueError("pulses must fit signed 16-bit")
        self._write_p4(P4.HOMING_TRIGGER_MODE, 0)
        self._write_p4(P4.HOMING_DIRECTION_SOURCE, source)
        self._write_p4(P4.HOMING_FINAL_MOVE, final)
        self._write_p4(P4.HOMING_STOP_MODE, stop)
        self._write_p4(P4.HOMING_HIGH_SPEED, high)
        self._write_p4(P4.HOMING_LOW_SPEED, low)
        self._write_p4(P4.HOMING_OFFSET_TURNS, turns)
        self._write_p4(P4.HOMING_OFFSET_PULSES, pulses)
        self._write_p4(P4.HOMING_TRIGGER_MODE, 2)

    def home(
        self,
        *,
        timeout: float,
        sensor: bool,
        confirm: str,
    ) -> MotionResult:
        """Run the already configured homing sequence through virtual SHOM."""
        self._require_confirmation(confirm, "HOME")
        self._require_io()
        self._validate_duration(timeout)
        direction_source = self._read_p4(P4.HOMING_DIRECTION_SOURCE)
        trigger_mode = self._read_p4(P4.HOMING_TRIGGER_MODE)
        high_speed = self._read_p4(P4.HOMING_HIGH_SPEED)
        low_speed = self._read_p4(P4.HOMING_LOW_SPEED)
        if trigger_mode != 2:
            raise ConfigError("P4-34 must be 2 for SHOM-triggered homing")
        if direction_source < 4 and not sensor:
            raise SafetyError(
                "limit/home sensor homing requires sensor=True"
            )
        self._validate_speed(high_speed, allow_zero=False)
        self._validate_speed(low_speed, allow_zero=False)
        self._validate_overspeed(
            high_speed, self._read_pa(PA.MAXIMUM_SPEED)
        )
        start_status = self.check(stopped=True).status
        self._require_mode(start_status, ControlMode.POSITION)
        self._require_position(start_status)
        restore = self._snapshot_pa(PA.POSITION_INPUT_MODE)
        samples = []
        try:
            self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
            self._write_pa(PA.POSITION_INPUT_MODE, PositionInputMode.INTERNAL, temporary=True)
            self._enable(confirm="ENABLE")
            self._write_p3(P3.VIRTUAL_INPUT_STATE, int(VirtualInput.START_HOMING))
            self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
            started = time.monotonic()
            motion_observed = False
            stopped_samples = 0
            while time.monotonic() - started < timeout:
                status = self.status()
                samples.append(status)
                if status.alarm_code:
                    code = status.alarm_code
                    raise RotatorError(
                        f"drive alarm during homing: "
                        f"{code} ({describe_alarm(code)})"
                    )
                motion_observed = motion_observed or any(
                    (
                        abs(status.motor_speed_rpm) > self.limits.threshold,
                        status.position_pulses != start_status.position_pulses,
                        status.position_command_pulses
                        != start_status.position_command_pulses,
                    )
                )
                if motion_observed and abs(status.motor_speed_rpm) <= self.limits.threshold:
                    stopped_samples += 1
                    if stopped_samples >= 3:
                        break
                else:
                    stopped_samples = 0
                time.sleep(0.1)
            else:
                raise SafetyError("homing timed out; use the hardware power cut if needed")
            motion_elapsed = time.monotonic() - started
        finally:
            self._write_p3(P3.VIRTUAL_INPUT_STATE, 0)
            try:
                self._wait_stopped()
            finally:
                self._restore_pa(restore)
        return MotionResult(
            0,
            timeout,
            tuple(samples),
            self.status(),
            motion_elapsed,
        )

    def _wait_stopped(self) -> RotatorStatus:
        deadline = time.monotonic() + self.limits.timeout
        last = self.status()
        while time.monotonic() < deadline:
            if abs(last.motor_speed_rpm) <= self.limits.threshold:
                return last
            time.sleep(0.1)
            last = self.status()
        raise SafetyError(
            f"motor did not stop within {self.limits.timeout}s; "
            "use the hardware power cut/E-stop"
        )

    def _read_one(self, address: int) -> int:
        values = self.client.read(address, 1)
        if len(values) != 1:
            raise RotatorError(f"register 0x{address:04X} returned {len(values)} values")
        return values[0]

    @staticmethod
    def _decode_i64(words: list[int]) -> int:
        if len(words) != 4:
            raise RotatorError("64-bit status value requires four registers")
        value = 0
        for index, word in enumerate(words):
            if not 0 <= word <= 0xFFFF:
                raise RotatorError("status register value is outside 16-bit range")
            value |= word << (index * 16)
        return value - (1 << 64) if value & (1 << 63) else value

    def _write_pa(self, number: int | PA, value: int, *, temporary: bool) -> None:
        parameter = int(number)
        self._write_verified(
            pa(parameter, temporary=temporary),
            self._word(value),
            verify_address=pa(parameter),
        )
        if not temporary:
            time.sleep(0.05)

    def _write_p3(self, number: int | P3, value: int) -> None:
        self._write_verified(p3(int(number)), self._word(value))

    def _write_p4(self, number: int | P4, value: int) -> None:
        self._write_verified(p4(int(number)), self._word(value))

    def _write_verified(
        self, address: int, value: int, *, verify_address: int | None = None
    ) -> None:
        self.client.write(address, value)
        actual = self._read_one(address if verify_address is None else verify_address)
        if actual != value:
            raise RotatorError(
                f"write verification failed after writing 0x{address:04X}: "
                f"wrote {value}, read {actual}"
            )

    def _snapshot_pa(self, *parameters: PA) -> dict[PA, int]:
        return {parameter: self._read_pa(parameter) for parameter in parameters}

    def _restore_pa(self, values: dict[PA, int]) -> None:
        for parameter, value in values.items():
            self._write_pa(parameter, value, temporary=True)

    def _require_io(self) -> None:
        if not self._io_ready():
            raise ConfigError(
                "virtual IO profile is not configured; call setup('position') first"
            )

    @staticmethod
    def _require_mode(status: RotatorStatus, expected: ControlMode) -> None:
        if status.control_mode != int(expected):
            raise ConfigError(
                f"active control mode is {status.control_mode}, expected {int(expected)}; "
                "call setup with the required mode first"
            )

    def _require_position(self, status: RotatorStatus) -> None:
        if (
            abs(status.position_error_pulses)
            > self.limits.error
        ):
            raise SafetyError(
                f"pending position error is {status.position_error_pulses} pulses; "
                "restart while disabled before enabling position control"
            )

    def _validate_speed(self, rpm: int, *, allow_zero: bool) -> None:
        if not allow_zero and rpm == 0:
            raise SafetyError("speed must not be zero")
        if abs(rpm) > self.limits.speed:
            raise SafetyError(
                f"speed {rpm} rpm exceeds safety limit {self.limits.speed} rpm"
            )

    def _validate_limit(self, percent: int) -> None:
        if not 1 <= percent <= self.limits.torque:
            raise SafetyError(
                f"torque limit must be 1..{self.limits.torque}%"
            )

    def _validate_overspeed(
        self, command_rpm: int, overspeed_limit_rpm: int
    ) -> None:
        minimum = command_rpm + self.limits.margin
        if overspeed_limit_rpm < minimum:
            raise SafetyError(
                f"PA23 overspeed limit must be at least {minimum} rpm for a "
                f"{command_rpm} rpm command"
            )

    def _validate_torque(self, percent: int) -> None:
        if abs(percent) > self.limits.torque:
            raise SafetyError(
                f"torque {percent}% exceeds safety limit "
                f"{self.limits.torque}%"
            )

    def _validate_duration(self, seconds: float) -> None:
        if not 0 < seconds <= self.limits.duration:
            raise SafetyError(
                f"duration must be >0 and <= {self.limits.duration}s"
            )

    @staticmethod
    def _require_confirmation(actual: str | None, expected: str) -> None:
        if actual != expected:
            raise SafetyError(f"operation requires confirm={expected!r}")

    @staticmethod
    def _word(value: int) -> int:
        integer = int(value)
        if integer < 0:
            return encode_i16(integer)
        if not 0 <= integer <= 0xFFFF:
            raise ValueError("register value must fit 16 bits")
        return integer


class Rotator:
    """Motorized rotator interface using degrees and degrees per second"""

    def __init__(
        self,
        client: RegisterClient,
        *,
        ratio: float = 180,
        speed: float = 30,
        torque: int = 15,
        ramp: int = 500,
        timeout: float = 60,
        device: int = 1,
        baudrate: int = 9600,
        format: int = 0,
    ) -> None:
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("ratio must be finite and positive")
        if not math.isfinite(speed) or not 0 < speed <= 30:
            raise ValueError("speed must be >0 and <=30 degrees/second")
        if not 1 <= torque <= 300:
            raise ValueError("torque must be 1..300")
        if not 1 <= ramp <= 10000:
            raise ValueError("ramp must be 1..10000")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.ratio = ratio
        self.speed = speed
        self.torque = torque
        self.ramp = ramp
        self.timeout = timeout
        self._homed = False
        self._motor = math.ceil(speed * ratio / 6.0)
        self._driver = _Controller(
            client,
            device=device,
            baudrate=baudrate,
            format=format,
            limits=SafetyLimits(
                speed=self._motor,
                torque=torque,
                duration=timeout,
                ramp=ramp,
                margin=0,
            ),
        )

    def status(self) -> RotatorStatus:
        return self._driver.status()

    def check(self, *, stopped: bool = True, alarm: bool = False) -> CheckResult:
        return self._driver.check(stopped=stopped, alarm=alarm)

    def setup(self, mode: str, *, confirm: str) -> RotatorStatus:
        if confirm != "SETUP":
            raise SafetyError("setup requires confirm='SETUP'")
        modes = {
            "position": ControlMode.POSITION,
            "speed": ControlMode.SPEED,
        }
        if mode not in modes:
            raise ValueError("mode must be 'position' or 'speed'")
        if mode == "position":
            self._driver.io(profile="motion", confirm="PERSIST")
        self._driver.mode(
            modes[mode],
            speed=self._motor,
            torque=self.torque,
            ramp=self.ramp,
            confirm="PERSIST",
        )
        status = self._driver.restart(confirm="RESTART")
        self._homed = False
        return status

    def enable(self, *, confirm: str) -> RotatorStatus:
        return self._driver._enable(confirm=confirm)

    def disable(self) -> RotatorStatus:
        return self._driver.disable()

    def stop(self) -> RotatorStatus:
        return self._driver.stop()

    def clear(self, *, confirm: str) -> RotatorStatus:
        return self._driver.clear(confirm=confirm)

    def calculate(self, *, angle: float, speed: float) -> Calculation:
        return calculate(
            angle=angle,
            speed=self._check_speed(speed),
            ratio=self.ratio,
            resolution=self._resolution(),
            ramp=self.ramp,
        )

    def move(
        self,
        *,
        angle: float,
        speed: float,
        slot: int = 1,
        torque: int | None = None,
        confirm: str,
    ) -> MoveResult:
        if angle == 0:
            raise ValueError("angle must not be zero")
        calculation = self.calculate(angle=angle, speed=speed)
        return self._position(
            calculation,
            turns=calculation.turns,
            pulses=calculation.pulses,
            mode=PositionCommandMode.INCREMENTAL,
            slot=slot,
            torque=torque,
            confirm=confirm,
        )

    def goto(
        self,
        *,
        angle: float,
        speed: float,
        slot: int = 1,
        torque: int | None = None,
        confirm: str,
    ) -> MoveResult:
        if not self._homed:
            raise ConfigError("goto requires a successful home in this session")
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        resolution = self._resolution()
        target = round(angle * self.ratio * resolution / 360.0)
        current = self.status().position_pulses
        delta = (target - current) * 360.0 / (self.ratio * resolution)
        calculation = calculate(
            angle=delta,
            speed=self._check_speed(speed),
            ratio=self.ratio,
            resolution=resolution,
            ramp=self.ramp,
        )
        if calculation.commands == 0:
            status = self.status()
            return MoveResult(calculation, 0.0, 0.0, 0, 0, status)
        turns = math.trunc(target / resolution)
        pulses = target - turns * resolution
        return self._position(
            calculation,
            turns=turns,
            pulses=pulses,
            mode=PositionCommandMode.ABSOLUTE_MULTITURN,
            slot=slot,
            torque=torque,
            confirm=confirm,
        )

    def spin(
        self,
        *,
        speed: float,
        duration: float | None = None,
        confirm: str,
    ) -> RotatorStatus | MotionResult:
        if confirm != "SPIN":
            raise SafetyError("spin requires confirm='SPIN'")
        rpm = self._rpm(speed)
        if rpm == 0:
            raise ValueError("speed must not be zero")
        if duration is not None:
            self._driver._validate_duration(duration)
        report = self._driver.check(stopped=True)
        self._driver._require_mode(report.status, ControlMode.SPEED)
        self._driver._configure_speed(
            maximum_rpm=self._motor,
            torque_limit_percent=self.torque,
            ramp_ms=self.ramp,
        )
        self._driver._enable(confirm="ENABLE")
        self._driver._set_speed(rpm, confirm="MOVE")
        if duration is None:
            return self.status()

        samples = []
        started = time.monotonic()
        try:
            while time.monotonic() - started < duration:
                status = self.status()
                samples.append(status)
                if status.alarm_code:
                    code = status.alarm_code
                    raise RotatorError(
                        f"drive alarm during speed motion: "
                        f"{code} ({describe_alarm(code)})"
                    )
                time.sleep(min(0.1, duration / 4))
        finally:
            final = self.stop()
        return MotionResult(
            rpm,
            duration,
            tuple(samples),
            final,
            time.monotonic() - started,
        )

    def home(self, *, speed: float, confirm: str) -> MotionResult:
        if confirm != "HOME":
            raise SafetyError("home requires confirm='HOME'")
        rpm = self._rpm(speed)
        if rpm == 0:
            raise ValueError("speed must not be zero")
        self._driver.homing(
            source=4 if rpm > 0 else 5,
            final=2,
            high=abs(rpm),
            low=max(1, min(500, abs(rpm) // 2)),
            confirm="PERSIST",
        )
        result = self._driver.home(
            timeout=self.timeout,
            sensor=False,
            confirm="HOME",
        )
        self._homed = True
        return result

    def _position(
        self,
        calculation: Calculation,
        *,
        turns: int,
        pulses: int,
        mode: PositionCommandMode,
        slot: int,
        torque: int | None,
        confirm: str,
    ) -> MoveResult:
        if confirm != "MOVE":
            raise SafetyError("position motion requires confirm='MOVE'")
        torque = self.torque if torque is None else torque
        self._driver._validate_limit(torque)
        if calculation.estimate + 3 > self.timeout:
            raise SafetyError(
                f"motion needs about {calculation.estimate:.2f}s but timeout is "
                f"{self.timeout:.2f}s"
            )
        report = self._driver.check(stopped=True)
        self._driver._require_mode(report.status, ControlMode.POSITION)
        self._driver._require_position(report.status)
        restore = self._driver._snapshot_pa(
            PA.MAXIMUM_SPEED,
            PA.ACCELERATION_TIME,
            PA.DECELERATION_TIME,
            PA.S_CURVE_TIME,
        )
        self._driver.slot(
            slot,
            turns=turns,
            pulses=pulses,
            rpm=calculation.rpm,
            mode=mode,
            confirm="PERSIST",
        )
        started = time.monotonic()
        try:
            self._driver._write_pa(
                PA.MAXIMUM_SPEED,
                calculation.rpm,
                temporary=True,
            )
            self._driver._write_pa(PA.ACCELERATION_TIME, self.ramp, temporary=True)
            self._driver._write_pa(PA.DECELERATION_TIME, self.ramp, temporary=True)
            self._driver._write_pa(PA.S_CURVE_TIME, 0, temporary=True)
            motion = self._driver.run(
                slot,
                timeout=calculation.estimate + 3,
                torque=torque,
                confirm="MOVE",
            )
        finally:
            self._driver._restore_pa(restore)
        return MoveResult(
            calculation=calculation,
            motion_seconds=motion.elapsed_seconds,
            operation_seconds=time.monotonic() - started,
            peak_speed_rpm=max(
                (abs(sample.motor_speed_rpm) for sample in motion.samples),
                default=0,
            ),
            peak_torque_percent=max(
                (abs(sample.motor_torque_percent) for sample in motion.samples),
                default=0,
            ),
            final_status=motion.final_status,
        )

    def _resolution(self) -> int:
        return self._driver._read_pa(PA.COMMAND_PULSES_PER_REVOLUTION)

    def _check_speed(self, speed: float) -> float:
        if not math.isfinite(speed) or not 0 < speed <= self.speed:
            raise SafetyError(
                f"speed must be >0 and <= {self.speed:g} degrees/second"
            )
        return speed

    def _rpm(self, speed: float) -> int:
        if not math.isfinite(speed) or speed == 0 or abs(speed) > self.speed:
            raise SafetyError(
                f"speed must be non-zero and within +/-{self.speed:g} degrees/second"
            )
        return int(math.copysign(math.ceil(abs(speed) * self.ratio / 6.0), speed))
