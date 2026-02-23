from celr.core.runtime import PersistentRuntime

def test_runtime():
    runtime = PersistentRuntime()
    print("Initializing Runtime...")

    # Step 1: Define a variable
    code1 = "x = 10\nprint(f'Defined x={x}')"
    output1, success1 = runtime.execute(code1)
    print(f"Step 1 Output: {output1.strip()} (Success: {success1})")

    # Step 2: Use the variable
    code2 = "y = x * 2\nprint(f'Calculated y={y}')"
    output2, success2 = runtime.execute(code2)
    print(f"Step 2 Output: {output2.strip()} (Success: {success2})")

    # Step 3: Check context
    print("\nContext Snapshot:")
    print(runtime.get_context_snapshot())

    if "y = 20" in output2:
        print("\n✅ TEST PASSED: State persisted!")
    else:
        print("\n❌ TEST FAILED: State lost.")

if __name__ == "__main__":
    test_runtime()
