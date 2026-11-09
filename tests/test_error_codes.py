"""错误码契约测试：保证「码必填、码表唯一、类与码不冲突」。

为什么要扫源码（AST）而不是只测运行时：
- code 现在是 raise 处的必填参数，漏传要等真跑进那条分支才暴露；
- 一个新写的 `raise KbNotFoundError("...")` 可能半年都不会被执行到，
  静态扫描能在提交时就拦住。

规则（与 app/core/exceptions.py 顶部约定一致）：
1. 所有业务异常实例的构造点必须带 `code=`（多数写在 `raise XxxError(...)` 里，
   少数先构造再由上层 `raise wrapped from e`，例如 IndexService._translate_chroma）；
2. code 必须写成 error_codes 里的具名常量，不允许裸数字（否则码表形同虚设）；
3. code 必须落在该异常类的白名单内 —— 这条把「类 → 允许的码」显式写下来，
   防止 `raise KbNotFoundError(code=ERR_KB_STORAGE)` 这种类名与码打架的写法；
4. 码表内所有码值唯一，且落在约定分段内。

运行：
    python tests/test_error_codes.py
    pytest tests/test_error_codes.py
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.core import error_codes  # noqa: E402

APP_DIR = os.path.join(ROOT, "app")
UTILS_DIR = os.path.join(ROOT, "utils")

# 类 → 该类允许出现的码。新增异常类或让某类承担新码时，先在这里登记。
ALLOWED_CODES = {
    "ParamError": {error_codes.CODE_ERROR},
    "KbNotFoundError": {error_codes.ERR_KB_NOT_FOUND},
    "KbStorageError": {
        error_codes.ERR_KB_STORAGE,
        error_codes.ERR_KB_DIM_MISMATCH,
        error_codes.ERR_KB_INDEX_BROKEN,
        error_codes.ERR_KB_STORAGE_FULL,
        error_codes.ERR_KB_VERSION_MISMATCH,
    },
    "GenerateError": {error_codes.ERR_GENERATE_FAILED},
    "UnsupportedFileTypeError": {error_codes.ERR_PARSE_UNSUPPORTED},
    "EmptyContentError": {error_codes.ERR_PARSE_EMPTY_CONTENT},
    "XlsxStyleBrokenError": {error_codes.ERR_PARSE_XLSX_BROKEN},
    # 解析链路的兜底类：凡是「这一页/这份文件没解析出东西」都走它，
    # 具体原因由 raise 处选 1xxx 段里的码表达。
    "DocumentParseError": {
        error_codes.ERR_PARSE_FAILED,
        error_codes.ERR_PARSE_OPEN_FAILED,
        error_codes.ERR_PARSE_ENCRYPTED,
        error_codes.ERR_PARSE_NO_PAGES,
        error_codes.ERR_PARSE_OCR_LIMIT,
        error_codes.ERR_PARSE_ALL_BLANK,
        error_codes.ERR_PARSE_EMPTY_CONTENT,
        error_codes.ERR_PARSE_XLSX_BROKEN,
    },
}

_CODE_NAMES = {
    name: value
    for name, value in vars(error_codes).items()
    if name.isupper() and isinstance(value, int) and not name.startswith("_")
}


def _iter_error_calls():
    """遍历 app/ 与 utils/ 下所有业务异常实例的构造点，产出 (文件, 行号, 类名, 调用节点)

    扫的是 `XxxError(...)` 调用本身，而不是外层 `raise`：有几处刻意先构造再上抛
    （如 IndexService._translate_chroma 返回实例、由 _guard 统一 `raise ... from e`），
    只认 raise 节点会把那些码误判成「定义了没人用」。
    """
    for base in (APP_DIR, UTILS_DIR):
        for dirpath, _, filenames in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in ALLOWED_CODES:
                        yield path, node.lineno, func.id, node


def _collect_code_arg(call: ast.Call):
    """取出 call 的 code 实参，返回 (值, 常量名)；无法静态求值时返回 (None, None)"""
    for kw in call.keywords:
        if kw.arg != "code":
            continue
        value = kw.value
        if isinstance(value, ast.Name):
            return _CODE_NAMES.get(value.id), value.id
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return value.value, None  # 裸数字：没有常量名
        return None, None
    return None, None


# ---------------------------------------------------------------- 码表本身

def test_code_values_unique():
    seen = {}
    for name, value in _CODE_NAMES.items():
        assert value not in seen, f"码值重复：{name}={value} 与 {seen[value]} 冲突"
        seen[value] = name


def test_code_segments():
    for name, value in _CODE_NAMES.items():
        if name.startswith("CODE_"):
            # Web 层：0 或标准 HTTP 状态码
            assert value == 0 or 400 <= value <= 599, f"{name}={value} 不在 Web 层分段"
        elif name.startswith("ERR_PARSE_"):
            assert 1000 <= value <= 1999, f"{name}={value} 不在 1xxx 解析段"
        elif name.startswith("ERR_KB_"):
            assert 2000 <= value <= 2999, f"{name}={value} 不在 2xxx 知识库段"
        elif name.startswith("ERR_GENERATE_"):
            assert 3000 <= value <= 3999, f"{name}={value} 不在 3xxx 生成段"
        else:
            raise AssertionError(f"错误码命名不符合 {name} 的约定前缀")


# ---------------------------------------------------------------- raise 用法

def test_every_raise_passes_code():
    missing = [
        f"{os.path.relpath(path, ROOT)}:{lineno} {cls} 漏传 code"
        for path, lineno, cls, call in _iter_error_calls()
        if _collect_code_arg(call)[0] is None and _collect_code_arg(call)[1] is None
    ]
    assert not missing, "以下位置未显式传 code：\n" + "\n".join(missing)


def test_raise_code_uses_named_constant():
    bare = [
        f"{os.path.relpath(path, ROOT)}:{lineno} {cls} 用了裸数字 code"
        for path, lineno, cls, call in _iter_error_calls()
        if _collect_code_arg(call)[1] is None
    ]
    assert not bare, "code 必须写成 error_codes 里的常量，不要硬编码数字：\n" + "\n".join(bare)


def test_raise_code_matches_class_whitelist():
    violations = []
    for path, lineno, cls, call in _iter_error_calls():
        value, name = _collect_code_arg(call)
        if value is None:
            continue  # 由上面两条用例负责报错
        allowed = ALLOWED_CODES[cls]
        if value not in allowed:
            violations.append(
                f"{os.path.relpath(path, ROOT)}:{lineno} {cls} 使用了 {name}={value}，"
                f"允许：{sorted(allowed)}"
            )
    assert not violations, "类与码不匹配：\n" + "\n".join(violations)


def test_every_definition_is_reachable():
    """码表里定义了的码，至少要出现在某个构造点的 code= 里（防止只增不用的死码）"""
    used = {_collect_code_arg(call)[0] for *_, call in _iter_error_calls()}
    # Web 层由 handler 直接产出，不经过 raise 自定义异常
    web_codes = {v for n, v in _CODE_NAMES.items() if n.startswith("CODE_")}
    dead = {n: v for n, v in _CODE_NAMES.items() if v not in used and v not in web_codes}
    assert not dead, f"以下码已定义但没有 raise 使用：{dead}"


if __name__ == "__main__":
    cases = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for case_name, fn in cases:
        try:
            fn()
            print(f"  PASS  {case_name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {case_name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {case_name}: {type(e).__name__}: {e}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
