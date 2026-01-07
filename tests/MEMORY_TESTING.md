# Memory Testing Guide

This document describes the memory safety testing capabilities in LibSVM's test suite.

## Overview

LibSVM uses multiple sanitizers to detect memory errors, undefined behavior, and other issues:

- **AddressSanitizer (ASAN)**: Detects memory errors
- **UndefinedBehaviorSanitizer (UBSan)**: Detects undefined behavior
- **Valgrind** (planned): Alternative memory checker for platforms without sanitizer support

## Sanitizers are Enabled by Default

Starting from recent updates, **sanitizers are enabled by default** for all test runs to maximize early detection of issues. This means:

- Unit tests run with ASAN + UBSan
- Integration tests run with ASAN + UBSan  
- Memory tests run with ASAN + UBSan (cannot be disabled)

## What ASAN Detects

AddressSanitizer detects:

- **Memory leaks**: Unreleased heap allocations
- **Heap buffer overflow**: Reading/writing beyond allocated memory
- **Stack buffer overflow**: Reading/writing beyond stack arrays
- **Use-after-free**: Accessing freed memory
- **Use-after-return**: Using stack memory after function returns
- **Double-free**: Freeing the same memory twice
- **Invalid pointer operations**: Comparing pointers from different allocations

### ASAN Configuration

Our ASAN configuration includes:

```bash
ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:allocator_may_return_null=1:\
check_initialization_order=1:detect_stack_use_after_return=1:\
strict_string_checks=1:detect_invalid_pointer_pairs=2:\
print_stats=1:atexit=1"
```

- `detect_leaks=1`: Enable memory leak detection
- `halt_on_error=0`: Continue running to find multiple errors
- `allocator_may_return_null=1`: Allow allocation failures for testing
- `check_initialization_order=1`: Detect initialization order bugs
- `detect_stack_use_after_return=1`: Detect stack use-after-return
- `strict_string_checks=1`: Stricter string operation checks
- `detect_invalid_pointer_pairs=2`: Check pointer comparisons
- `print_stats=1`: Print statistics after run
- `atexit=1`: Check for leaks at program exit

## What UBSan Detects

UndefinedBehaviorSanitizer detects:

- **Integer overflow**: Signed integer overflow
- **Division by zero**: Integer division by zero
- **Null pointer dereference**: Dereferencing NULL pointers
- **Misaligned access**: Unaligned memory accesses
- **Invalid enum values**: Out-of-range enum values
- **Invalid bool values**: Non-0/1 bool values
- **Type mismatches**: Casting between incompatible types

### UBSan Configuration

```bash
UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0"
```

- `print_stacktrace=1`: Print full stack traces for errors
- `halt_on_error=0`: Continue running to find multiple errors

## Running Tests with Sanitizers

### Default Behavior (Sanitizers Enabled)

```bash
# All tests with sanitizers
./scripts/run_tests.sh

# Specific categories
./scripts/run_tests.sh --unit
./scripts/run_tests.sh --integration
./scripts/run_tests.sh --memory      # Always uses sanitizers
```

### Disabling Sanitizers (Not Recommended)

```bash
# Only for performance testing or when sanitizers are unavailable
./scripts/run_tests.sh --no-sanitize
```

**Note**: Memory tests always run with sanitizers, even with `--no-sanitize`.

### Manual CMake Build

```bash
# Build with sanitizers
mkdir build && cd build
cmake -DLIBSVM_BUILD_TESTS=ON \
      -DLIBSVM_ENABLE_ASAN=ON \
      -DLIBSVM_ENABLE_UBSAN=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make

# Run tests
ASAN_OPTIONS="detect_leaks=1:..." UBSAN_OPTIONS="print_stacktrace=1:..." ctest
```

## GitHub Actions CI

All CI runs use sanitizers by default:

- Linux: ASAN + UBSan (GCC/Clang)
- macOS: ASAN + UBSan (Clang)
- Windows: No sanitizers (MSVC has limited support)

## Interpreting Results

### ASAN Error Example

```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014
READ of size 4 at 0x602000000014 thread T0
    #0 0x4007c8 in main example.cpp:6
    #1 0x7f8b7c7b182f in __libc_start_main
```

This indicates:
- **Error type**: heap-buffer-overflow
- **Location**: example.cpp line 6
- **Operation**: Read 4 bytes beyond allocated buffer

### UBSan Error Example

```
runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
    #0 0x4007d9 in main example.cpp:10
```

This indicates:
- **Error type**: Signed integer overflow
- **Location**: example.cpp line 10

### Memory Leak Example

```
=================================================================
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 100 byte(s) in 1 object(s) allocated from:
    #0 0x7f8b7c9b1b50 in malloc
    #1 0x4007c8 in main example.cpp:5
```

This indicates:
- **Leak size**: 100 bytes
- **Allocation site**: example.cpp line 5

## Best Practices

1. **Always run tests with sanitizers during development**
   - Catches issues early
   - Provides detailed error reports

2. **Fix all sanitizer errors before committing**
   - Even if tests pass, sanitizer errors indicate bugs
   - CI will fail if sanitizers detect issues

3. **Run memory tests separately for detailed checks**
   ```bash
   ./scripts/run_tests.sh --memory
   ```

4. **Use RelWithDebInfo build type**
   - Optimizations enabled for realistic performance
   - Debug info preserved for stack traces
   - This is the default when sanitizers are enabled

5. **Check CI logs for sanitizer output**
   - GitHub Actions preserves full sanitizer reports
   - Look for "ERROR: AddressSanitizer" or "runtime error"

## Platform Support

| Platform | ASAN | UBSan | Notes |
|----------|------|-------|-------|
| Linux (GCC) | ✅ | ✅ | Full support |
| Linux (Clang) | ✅ | ✅ | Full support |
| macOS (Clang) | ✅ | ✅ | Full support |
| Windows (MSVC) | ❌ | ❌ | Limited support, not enabled |
| Windows (MinGW) | ⚠️ | ⚠️ | Partial support |

## Performance Impact

Sanitizers have performance overhead:

- **ASAN**: 2-5x slowdown, 2-3x memory usage
- **UBSan**: ~20% slowdown, minimal memory overhead
- **Combined**: ~5-7x slowdown

This is acceptable for testing. For production builds, sanitizers are disabled.

## Future Enhancements

Planned additions:

1. **ThreadSanitizer (TSan)**: For detecting race conditions (when multi-threading is added)
2. **MemorySanitizer (MSan)**: For detecting use of uninitialized memory
3. **Valgrind integration**: Alternative for platforms without sanitizer support
4. **Continuous fuzzing**: Using LibFuzzer with sanitizers

## References

- [AddressSanitizer Documentation](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [UndefinedBehaviorSanitizer Documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [GoogleTest Documentation](https://google.github.io/googletest/)
