"""Public turntable API and its focused command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from .protocol import (
    ConfigError,
    Interface,
    InterfaceConfig,
    InterfaceError,
    SafetyError,
    RotatorError,
    ports,
    probe_ports,
)
from .control import (
    CheckResult,
    Calculation,
    MoveResult,
    MotionResult,
    Rotator,
    RotatorStatus,
    calculate,
)

__all__ = [
    "Interface",
    "Rotator",
    "InterfaceConfig",
    "InterfaceError",
    "SafetyError",
    "RotatorError",
    "ConfigError",
    "ports",
    "RotatorDetection",
    "detect_rotator",
    "Calculation",
    "MoveResult",
    "RotatorStatus",
    "MotionResult",
    "CheckResult",
    "calculate",
]


@dataclass(frozen=True, slots=True)
class RotatorDetection:
    """A uniquely identified drive and the read-only validation result."""

    port: str
    check: CheckResult

    def data(self) -> dict[str, object]:
        return {"port": self.port, **self.check.data()}


def detect_rotator(
    *,
    device: int = 1,
    baudrate: int = 9600,
    bytesize: int = 8,
    parity: str = "N",
    stopbits: int = 2,
    timeout: float = 0.5,
    retries: int = 0,
    candidate_ports: Sequence[str] | None = None,
) -> RotatorDetection:
    """Find exactly one compatible drive using read-only Modbus requests."""

    candidates = list(
        dict.fromkeys(candidate_ports if candidate_ports is not None else probe_ports())
    )
    if not candidates:
        raise InterfaceError("未找到可用串口")

    matches: list[RotatorDetection] = []
    failures: list[str] = []
    for port in candidates:
        try:
            config = InterfaceConfig(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout,
                retries=retries,
            )
            with Interface(config, device=device) as client:
                stage = Rotator(
                    client,
                    device=device,
                    baudrate=baudrate,
                    format=_serial_format(config),
                )
                # 识别过程不得改变驱动器状态
                # 报告报警和运动状态但不阻止识别已配置的驱动器
                report = stage.check(stopped=False, alarm=True)
            matches.append(RotatorDetection(port=port, check=report))
        except (InterfaceError, RotatorError, ValueError) as exc:
            failures.append(f"{port}: {exc}")

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(match.port for match in matches)
        raise InterfaceError(f"检测到多个匹配的转台串口 ({names})，请使用 --port 明确指定")
    details = "; ".join(failures)
    raise InterfaceError(f"未检测到匹配的转台；已尝试 {', '.join(candidates)}。{details}")


def _add_options(
    parser: argparse.ArgumentParser,
    *,
    timeout: float = 1.0,
    retries: int = 1,
) -> None:
    parser.add_argument(
        "--port",
        default=None,
        help="串口名称，例如 COM7；省略时自动探测",
    )
    parser.add_argument("--device", type=int, default=1, help="Modbus 从机地址")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--bytesize", type=int, default=8)
    parser.add_argument("--parity", choices=["N", "E", "O"], default="N")
    parser.add_argument("--stopbits", type=int, choices=[1, 2], default=2)
    parser.add_argument("--timeout", type=float, default=timeout)
    parser.add_argument("--retries", type=int, default=retries)


def _drive_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    command = commands.add_parser(name, help=help_text)
    _add_options(command)
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotator", description="电机驱动器与转台安全控制接口"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("ports", help="列出串口")
    detect = commands.add_parser("detect", help="只读探测并验证唯一的转台串口")
    _add_options(detect, timeout=0.5, retries=0)
    _drive_command(commands, "status", "读取实时运行状态")
    _drive_command(commands, "check", "检查通信、报警和停止状态")
    _drive_command(commands, "stop", "停止运动并保持使能")

    clear = _drive_command(commands, "clear", "停机后清除可复位报警")
    clear.add_argument("--confirm", required=True, help="必须填写 CLEAR")

    setup = _drive_command(commands, "setup", "安全写入位置/速度模式并软重启驱动")
    setup.add_argument("mode", choices=["position", "speed"])
    setup.add_argument("--ratio", type=float, default=180)
    setup.add_argument("--speed", type=float, default=30)
    setup.add_argument("--torque", type=int, default=15)
    setup.add_argument("--ramp", type=int, default=500)
    setup.add_argument("--confirm", required=True, help="必须填写 SETUP")

    calculation = commands.add_parser(
        "calculate", help="计算转台角度运动参数，不连接设备"
    )
    calculation.add_argument("angle", type=float, help="转台角度，可为负数")
    calculation.add_argument("--speed", type=float, required=True)
    calculation.add_argument("--ratio", type=float, required=True)
    calculation.add_argument("--resolution", type=int, default=1000)
    calculation.add_argument("--ramp", type=int, default=500)

    move = _drive_command(commands, "move", "按转台角度和速度执行位置运动")
    move.add_argument("angle", type=float, help="转台角度，可为负数")
    move.add_argument("--speed", type=float, required=True)
    move.add_argument("--ratio", type=float, default=180)
    move.add_argument("--ramp", type=int, default=500)
    move.add_argument("--slot", type=int, choices=range(1, 9), default=1)
    move.add_argument("--torque", type=int, default=15, help="临时转矩上限百分比")
    move.add_argument("--limit", type=float, default=60, help="单次运动最长秒数")
    move.add_argument("--confirm", required=True, help="必须填写 MOVE")
    return parser


def _config(args: argparse.Namespace) -> InterfaceConfig:
    port = args.port
    if port is None:
        port = detect_rotator(
            device=args.device,
            baudrate=args.baudrate,
            bytesize=args.bytesize,
            parity=args.parity,
            stopbits=args.stopbits,
        ).port
    return InterfaceConfig(
        port=port,
        baudrate=args.baudrate,
        bytesize=args.bytesize,
        parity=args.parity,
        stopbits=args.stopbits,
        timeout=args.timeout,
        retries=args.retries,
    )


def _serial_format(config: InterfaceConfig) -> int:
    formats = {
        (8, "N", 2): 0,
        (8, "E", 1): 1,
        (8, "O", 1): 2,
        (8, "N", 1): 3,
    }
    return formats[(config.bytesize, config.parity, config.stopbits)]


def _rotator(args: argparse.Namespace, client: Interface) -> Rotator:
    config = client.config
    return Rotator(
        client,
        ratio=getattr(args, "ratio", 180),
        speed=30,
        torque=getattr(args, "torque", 15),
        ramp=getattr(args, "ramp", 500),
        timeout=getattr(args, "limit", 60),
        device=args.device,
        baudrate=config.baudrate,
        format=_serial_format(config),
    )


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ports":
            found = ports()
            print("\n".join(found) if found else "未找到串口")
            return 0
        if args.command == "calculate":
            _print_json(
                calculate(
                    angle=args.angle,
                    speed=args.speed,
                    ratio=args.ratio,
                    resolution=args.resolution,
                    ramp=args.ramp,
                ).data()
            )
            return 0
        if args.command == "detect":
            _print_json(
                detect_rotator(
                    device=args.device,
                    baudrate=args.baudrate,
                    bytesize=args.bytesize,
                    parity=args.parity,
                    stopbits=args.stopbits,
                    timeout=args.timeout,
                    retries=args.retries,
                    candidate_ports=[args.port] if args.port else None,
                ).data()
            )
            return 0

        config = _config(args)
        with Interface(config, device=args.device) as client:
            rotator = _rotator(args, client)
            if args.command == "status":
                _print_json(rotator.status().data())
            elif args.command == "check":
                _print_json(rotator.check().data())
            elif args.command == "stop":
                _print_json(rotator.stop().data())
            elif args.command == "clear":
                _print_json(rotator.clear(confirm=args.confirm).data())
            elif args.command == "setup":
                _print_json(rotator.setup(args.mode, confirm=args.confirm).data())
            elif args.command == "move":
                try:
                    result = rotator.move(
                        angle=args.angle,
                        speed=args.speed,
                        slot=args.slot,
                        torque=args.torque,
                        confirm=args.confirm,
                    )
                    _print_json(result.data())
                finally:
                    rotator.disable()
    except (InterfaceError, RotatorError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
