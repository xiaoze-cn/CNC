"""Serial transport, register map, codecs, and shared errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any, Literal

from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
from serial import SerialException
from serial.tools import list_ports


class InterfaceError(Exception):
    """Serial communication or Modbus response failure."""


class RotatorError(Exception):
    """Rotator control operation failure."""


class SafetyError(RotatorError):
    """Operation rejected by a motion safety check."""


class ConfigError(RotatorError):
    """Drive configuration does not match the requested operation."""


@dataclass(frozen=True, slots=True)
class InterfaceConfig:
    """Serial settings for the connected motor drive."""

    port: str
    baudrate: int = 9600
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 2
    timeout: float = 1.0
    retries: int = 1

    def __post_init__(self) -> None:
        if not self.port.strip():
            raise ValueError("串口名称不能为空")
        if not 4800 <= self.baudrate <= 115200 or self.baudrate % 100 != 0:
            raise ValueError("波特率必须是 4800 到 115200 之间的整百数")
        serial_format = (self.bytesize, self.parity.upper(), self.stopbits)
        if serial_format not in {
            (8, "N", 2),
            (8, "E", 1),
            (8, "O", 1),
            (8, "N", 1),
        }:
            raise ValueError("串口格式必须是手册支持的 8N2、8E1、8O1 或 8N1")
        if self.timeout <= 0:
            raise ValueError("超时时间必须大于零")
        if self.retries < 0:
            raise ValueError("重试次数不能小于零")
        object.__setattr__(self, "parity", self.parity.upper())


WordOrder = Literal["big", "little"]


def encode_i16(value: int) -> int:
    if not -(1 << 15) <= value < (1 << 15):
        raise ValueError("数值必须在有符号 16 位整数范围内")
    return value & 0xFFFF


def decode_i16(register: int) -> int:
    _validate_register(register)
    return register - 0x10000 if register & 0x8000 else register


def encode_u32(value: int, word_order: WordOrder = "big") -> list[int]:
    if not 0 <= value < (1 << 32):
        raise ValueError("数值必须在无符号 32 位整数范围内")
    return _order_words([(value >> 16) & 0xFFFF, value & 0xFFFF], word_order)


def decode_u32(
    registers: list[int] | tuple[int, int], word_order: WordOrder = "big"
) -> int:
    high, low = _normalized_words(registers, word_order)
    return (high << 16) | low


def encode_i32(value: int, word_order: WordOrder = "big") -> list[int]:
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError("数值必须在有符号 32 位整数范围内")
    return encode_u32(value & 0xFFFFFFFF, word_order)


def decode_i32(
    registers: list[int] | tuple[int, int], word_order: WordOrder = "big"
) -> int:
    value = decode_u32(registers, word_order)
    return value - 0x100000000 if value & 0x80000000 else value


def _validate_register(register: int) -> None:
    if not 0 <= register <= 0xFFFF:
        raise ValueError("寄存器数值必须在 0 到 65535 之间")


def _order_words(words: list[int], word_order: WordOrder) -> list[int]:
    if word_order == "big":
        return words
    if word_order == "little":
        return list(reversed(words))
    raise ValueError("字序必须是 big 或 little")


def _normalized_words(
    registers: list[int] | tuple[int, int], word_order: WordOrder
) -> tuple[int, int]:
    if len(registers) != 2:
        raise ValueError("必须提供两个寄存器")
    words = _order_words(list(registers), word_order)
    _validate_register(words[0])
    _validate_register(words[1])
    return words[0], words[1]


ALARM_DESCRIPTIONS = {
    0: "正常",
    1: "超速",
    2: "主电路过压",
    3: "主电路欠压",
    4: "位置超差",
    5: "驱动器过热",
    6: "速度放大器饱和",
    7: "驱动禁止异常",
    8: "位置偏差计数器溢出",
    11: "IPM 模块故障",
    13: "驱动器或电机过载",
    14: "制动故障",
    18: "继电器开关故障",
    19: "抱闸延时错误",
    20: "EEPROM 错误",
    21: "FPGA 模块故障",
    23: "电流采集电路故障",
    29: "用户转矩过载",
    38: "编码器 EEPROM 通信失败",
    39: "编码器数据 CRC 校验错误",
    40: "不支持的电机型号",
    41: "需要切换电机型号",
    42: "AC 输入电压过低",
    47: "上电时主电路电压过高",
    55: "RS485 通信地址错误",
    56: "RS485 通信数据错误",
    57: "RS485 通信超时",
}


def describe_alarm(code: int) -> str:
    return ALARM_DESCRIPTIONS.get(code, "未知报警")


PA_TEMPORARY_OFFSET = 0x0080
STATUS_START = 0x1000
STATUS_COUNT = 0x1C


def pa(number: int, *, temporary: bool = False) -> int:
    if not 0 <= number <= 105:
        raise ValueError("PA parameter number must be between 0 and 105")
    return number + (PA_TEMPORARY_OFFSET if temporary else 0)


def p3(number: int) -> int:
    if not 0 <= number <= 45:
        raise ValueError("P3 parameter number must be between 0 and 45")
    return 0x0100 + number


def p4(number: int) -> int:
    if not 0 <= number <= 39:
        raise ValueError("P4 parameter number must be between 0 and 39")
    return 0x0200 + number


class PA(IntEnum):
    MODEL_CODE = 1
    SOFTWARE_VERSION = 2
    CONTROL_MODE = 4
    POSITION_GAIN = 9
    COMMAND_PULSES_PER_REVOLUTION = 11
    ELECTRONIC_GEAR_NUMERATOR = 12
    ELECTRONIC_GEAR_DENOMINATOR = 13
    POSITION_INPUT_MODE = 14
    POSITION_DIRECTION_INVERT = 15
    POSITION_COMPLETE_RANGE = 16
    POSITION_ERROR_RANGE = 17
    POSITION_ERROR_DISABLE = 18
    POSITION_SMOOTHING = 19
    DIRECTION_LIMIT_DISABLE = 20
    JOG_SPEED = 21
    SPEED_SOURCE = 22
    MAXIMUM_SPEED = 23
    INTERNAL_SPEED_1 = 24
    INTERNAL_SPEED_2 = 25
    INTERNAL_SPEED_3 = 26
    INTERNAL_SPEED_4 = 27
    TORQUE_SOURCE = 32
    INTERNAL_CCW_TORQUE_LIMIT = 34
    INTERNAL_CW_TORQUE_LIMIT = 35
    ACCELERATION_TIME = 40
    DECELERATION_TIME = 41
    S_CURVE_TIME = 42
    TORQUE_MODE_SPEED_LIMIT = 50
    FORCE_ENABLE = 53
    SOFT_RESET = 60
    ALARM_CLEAR = 61
    INTERNAL_TORQUE_1 = 64
    INTERNAL_TORQUE_2 = 65
    INTERNAL_TORQUE_3 = 66
    INTERNAL_TORQUE_4 = 67
    MODBUS_ADDRESS = 71
    MODBUS_BAUDRATE = 72
    MODBUS_FORMAT = 73
    COMMUNICATION_ERROR_ACTION = 74
    RS485_ENABLE = 104


class P3(IntEnum):
    FORCE_INPUTS_1 = 15
    FORCE_INPUTS_2 = 16
    FORCE_INPUTS_3 = 17
    FORCE_INPUTS_4 = 18
    FORCE_INPUTS_5 = 19
    VIRTUAL_IO_MODE = 30
    VIRTUAL_INPUT_STATE = 31
    VIRTUAL_DI1_FUNCTION = 38
    VIRTUAL_DI2_FUNCTION = 39
    VIRTUAL_DI3_FUNCTION = 40
    VIRTUAL_DI4_FUNCTION = 41
    VIRTUAL_DI5_FUNCTION = 42
    VIRTUAL_DI6_FUNCTION = 43
    VIRTUAL_DI7_FUNCTION = 44
    VIRTUAL_DI8_FUNCTION = 45


class P4(IntEnum):
    POSITION_COMMAND_MODE = 0
    COMMAND_COMPLETE_DELAY = 1
    HOMING_DIRECTION_SOURCE = 32
    HOMING_FINAL_MOVE = 33
    HOMING_TRIGGER_MODE = 34
    HOMING_STOP_MODE = 35
    HOMING_HIGH_SPEED = 36
    HOMING_LOW_SPEED = 37
    HOMING_OFFSET_TURNS = 38
    HOMING_OFFSET_PULSES = 39


class StatusRegister(IntEnum):
    MOTOR_SPEED = 0x1000
    POSITION_LOW = 0x1001
    POSITION_HIGH = 0x1002
    POSITION_COMMAND_LOW = 0x1003
    POSITION_COMMAND_HIGH = 0x1004
    POSITION_ERROR_LOW = 0x1005
    POSITION_ERROR_HIGH = 0x1006
    MOTOR_TORQUE = 0x1007
    MOTOR_CURRENT = 0x1008
    CONTROL_MODE = 0x1009
    TEMPERATURE = 0x100A
    SPEED_COMMAND = 0x100B
    TORQUE_COMMAND = 0x100C
    INPUT_STATE = 0x100F
    OUTPUT_STATE = 0x1010
    BUS_VOLTAGE = 0x1012
    ALARM_CODE = 0x1013
    RELAY_STATE = 0x1015
    RUN_STATE = 0x1016
    EXTERNAL_VOLTAGE_STATE = 0x1017
    ABSOLUTE_POSITION_WORD_0 = 0x1018
    ABSOLUTE_POSITION_WORD_1 = 0x1019
    ABSOLUTE_POSITION_WORD_2 = 0x101A
    ABSOLUTE_POSITION_WORD_3 = 0x101B


class ControlMode(IntEnum):
    POSITION = 0
    SPEED = 1
    TORQUE = 2
    POSITION_SPEED = 3
    POSITION_TORQUE = 4
    SPEED_TORQUE = 5


class SpeedSource(IntEnum):
    ANALOG = 0
    INTERNAL = 1
    ANALOG_AND_INTERNAL = 2
    JOG = 3
    KEYPAD = 4
    IO_JOG = 5


class TorqueSource(IntEnum):
    ANALOG = 0
    INTERNAL = 1
    ANALOG_AND_INTERNAL = 2


class PositionInputMode(IntEnum):
    PULSE_AND_DIRECTION = 0
    CW_CCW_PULSES = 1
    AB_QUADRATURE = 2
    INTERNAL = 3


class PositionCommandMode(IntEnum):
    ABSOLUTE_MULTITURN = 0
    INCREMENTAL = 1
    ABSOLUTE_SINGLETURN = 2


class DigitalInputFunction(IntEnum):
    NONE = 0
    DRIVE_ENABLE = 1
    ALARM_RESET = 2
    COMMAND_INVERT = 9
    MODE_SELECT = 16
    JOG_POSITIVE = 22
    JOG_NEGATIVE = 23
    POSITION_HOLD = 27
    POSITION_TRIGGER = 28
    POSITION_SELECT_0 = 29
    POSITION_SELECT_1 = 30
    POSITION_SELECT_2 = 31
    START_HOMING = 33
    HOME_SENSOR = 34


class VirtualInput(IntFlag):
    JOG_POSITIVE = 1 << 0
    JOG_NEGATIVE = 1 << 1
    POSITION_TRIGGER = 1 << 2
    POSITION_SELECT_0 = 1 << 3
    POSITION_SELECT_1 = 1 << 4
    POSITION_SELECT_2 = 1 << 5
    POSITION_HOLD = 1 << 6
    MODE_SELECT = 1 << 6
    START_HOMING = 1 << 7


VIRTUAL_IO_PROFILE = (
    DigitalInputFunction.JOG_POSITIVE,
    DigitalInputFunction.JOG_NEGATIVE,
    DigitalInputFunction.POSITION_TRIGGER,
    DigitalInputFunction.POSITION_SELECT_0,
    DigitalInputFunction.POSITION_SELECT_1,
    DigitalInputFunction.POSITION_SELECT_2,
    DigitalInputFunction.POSITION_HOLD,
    DigitalInputFunction.START_HOMING,
)

VIRTUAL_IO_MIXED_PROFILE = (
    DigitalInputFunction.JOG_POSITIVE,
    DigitalInputFunction.JOG_NEGATIVE,
    DigitalInputFunction.POSITION_TRIGGER,
    DigitalInputFunction.POSITION_SELECT_0,
    DigitalInputFunction.POSITION_SELECT_1,
    DigitalInputFunction.POSITION_SELECT_2,
    DigitalInputFunction.MODE_SELECT,
    DigitalInputFunction.START_HOMING,
)


def slot_registers(slot: int) -> tuple[int, int, int]:
    if not 1 <= slot <= 8:
        raise ValueError("position slot must be between 1 and 8")
    first = 2 + (slot - 1) * 3
    return p4(first), p4(first + 1), p4(first + 2)


class Interface:
    """Synchronous Modbus RTU client with strict response validation."""

    def __init__(self, config: InterfaceConfig, device: int = 1) -> None:
        if not 1 <= device <= 254:
            raise ValueError("从机地址必须在 1 到 254 之间")
        self.config = config
        self.device = device
        self._client = ModbusSerialClient(
            port=config.port,
            baudrate=config.baudrate,
            bytesize=config.bytesize,
            parity=config.parity,
            stopbits=config.stopbits,
            timeout=config.timeout,
            retries=config.retries,
        )

    @property
    def connected(self) -> bool:
        return self._client.connected

    def connect(self) -> None:
        try:
            if not self._client.connect():
                raise InterfaceError(f"无法打开串口 {self.config.port}")
        except (OSError, SerialException, ModbusException) as exc:
            raise InterfaceError(f"无法打开串口 {self.config.port}: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Interface:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read(
        self, address: int, count: int = 1, *, kind: str = "holding"
    ) -> list[int]:
        self._validate_range(address, count, maximum_count=125)
        self._ensure_connected()
        methods = {
            "holding": self._client.read_holding_registers,
            "input": self._client.read_input_registers,
        }
        if kind not in methods:
            raise ValueError("kind must be 'holding' or 'input'")
        return self._registers(
            self._call(
                "读取寄存器",
                methods[kind],
                address=address,
                count=count,
                device_id=self.device,
            ),
            expected_count=count,
        )

    def write(self, address: int, value: int | Sequence[int]) -> None:
        self._ensure_connected()
        if isinstance(value, int):
            self._validate_range(address, 1, maximum_count=1)
            self._validate_word(value)
            self._call(
                "写入单个寄存器",
                self._client.write_register,
                address=address,
                value=value,
                device_id=self.device,
            )
            return
        words = list(value)
        self._validate_range(address, len(words), maximum_count=123)
        for word in words:
            self._validate_word(word)
        self._call(
            "写入多个寄存器",
            self._client.write_registers,
            address=address,
            values=words,
            device_id=self.device,
        )

    def _ensure_connected(self) -> None:
        if not self.connected:
            self.connect()

    def _call(self, operation: str, method: Any, **kwargs: Any) -> Any:
        try:
            response = method(**kwargs)
        except (OSError, SerialException, ModbusException) as exc:
            raise InterfaceError(f"{operation}失败: {exc}") from exc
        if response is None:
            raise InterfaceError(f"{operation}失败: 未收到 Modbus 响应")
        if response.isError():
            raise InterfaceError(f"{operation}失败: {response}")
        return response

    @staticmethod
    def _registers(response: Any, *, expected_count: int) -> list[int]:
        registers = getattr(response, "registers", None)
        if registers is None:
            raise InterfaceError("Modbus 响应中没有寄存器数据")
        result = list(registers)
        if len(result) != expected_count:
            raise InterfaceError(
                f"Modbus 返回 {len(result)} 个寄存器，预期 {expected_count} 个"
            )
        return result

    @staticmethod
    def _validate_range(address: int, count: int, *, maximum_count: int) -> None:
        if not 0 <= address <= 0xFFFF:
            raise ValueError("寄存器地址必须在 0x0000 到 0xFFFF 之间")
        if not 1 <= count <= maximum_count:
            raise ValueError(f"寄存器数量必须在 1 到 {maximum_count} 之间")
        if address + count - 1 > 0xFFFF:
            raise ValueError("寄存器范围超出 0xFFFF")

    @staticmethod
    def _validate_word(value: int) -> None:
        if not 0 <= value <= 0xFFFF:
            raise ValueError("寄存器数值必须在 0 到 65535 之间")


def ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def probe_ports() -> list[str]:
    """Return ports safe to open during unattended probing.

    Opening an unconnected Windows Bluetooth SPP port may block well beyond the
    configured serial timeout, so those virtual ports are left for explicit use.
    """

    return [
        port.device
        for port in list_ports.comports()
        if "BTHENUM" not in (port.hwid or "").upper()
    ]
