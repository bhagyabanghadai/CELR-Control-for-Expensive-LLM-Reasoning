
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

try:
    from celr.core.executor import TaskExecutor
    from celr.core.types import TaskContext, Plan, Step, StepType
    from celr.cortex.policy import CortexAction, MetaPolicy
    from celr.cortex.state import StateExtractor
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# Override print to ensure logging
original_print = print
def print(*args, **kwargs):
    msg = " ".join(map(str, args))
    with open("debug_log.txt", "a") as f:
        f.write(msg + "\n")
    original_print(*args, **kwargs)

def log(msg):
    with open("debug_log.txt", "a") as f:
        f.write(str(msg) + "\n")
    print(msg)

def test_manual():
    # clear log
    with open("debug_log.txt", "w") as f:
        f.write("Starting log\n")

    try:
        log("Starting manual verification...")
        
        # Setup Context
        context = MagicMock(spec=TaskContext)
        context.budget_limit = 10.0
        context.current_spread_usd = 0.0
        context.user_prompt = "Write python code"
        context.cost_tracker = MagicMock()
        context.cost_tracker.total_cost = 0.0
        context.cost_tracker.can_afford.return_value = True
        context.log = MagicMock()

        # Mocks
        planner = MagicMock()
        cost_tracker = context.cost_tracker
        escalation = MagicMock()
        escalation.get_provider.return_value = MagicMock()
        tools = MagicMock()
        verifier = MagicMock()
        verifier.verify.return_value = True
        reflection = MagicMock()

        # Init Executor
        log("Initializing TaskExecutor...")
        executor = TaskExecutor(
            context, planner, cost_tracker, escalation, tools, verifier, reflection
        )

        # Mock Cortex components
        mock_extractor = MagicMock(spec=StateExtractor)
        mock_policy = MagicMock(spec=MetaPolicy)
        
        executor.state_extractor = mock_extractor
        executor.policy = mock_policy
        
        # Set expected action
        mock_policy.get_action.return_value = CortexAction.ESCALATE
        log(f"Mock Policy Action set to: {CortexAction.ESCALATE}")

        # Create Plan
        log("Creating Step...")
        step = Step(id="1", description="test step", step_type=StepType.REASONING)
        step.max_retries = 0
        step.status = "PENDING"
        step.estimated_difficulty = 0.5
        
        log("Creating Plan...")
        plan = Plan(items=[step], original_goal="Test Goal")
        planner.get_ready_steps.side_effect = [[step], []]
        
        # Run
        log("Running executor...")
        executor.run(plan)
        log("Executor run finished.")

        # Verify calls
        log("Verifying calls...")
        mock_extractor.extract.assert_called()
        log("StateExtractor.extract called successfully.")
        
        mock_policy.get_action.assert_called()
        log("MetaPolicy.get_action called successfully.")
        
        # Verify Force Prompt
        escalation.get_provider.assert_called()
        call_args = escalation.get_provider.call_args
        log(f"Escalation call args: {call_args}")
        
        if 'force_expensive' in call_args.kwargs:
            val = call_args.kwargs['force_expensive']
            if val is True:
                log("SUCCESS: force_expensive=True passed to escalation.")
            else:
                log(f"FAILURE: force_expensive={val}")
                sys.exit(1)
        else:
            # Check positional args
             if len(call_args.args) > 1 and call_args.args[1] is True:
                 log("SUCCESS: force_expensive=True passed as positional arg.")
             else:
                 log("FAILURE: force_expensive not found at all")
                 sys.exit(1)

    except Exception as e:
        import traceback
        log(f"CRITICAL ERROR: {e}")
        log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    test_manual()
