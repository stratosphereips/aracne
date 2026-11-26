import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from lib import planner


class PlannerResponseTests(unittest.TestCase):
    def test_incomplete_plan_can_have_no_actions(self):
        response = json.dumps({
            "steps": [],
            "goal_reached": False,
            "goal_verification": "A target address is required before testing.",
        })
        plan = planner.parse_plan_response(response, caller="generate_plan")
        self.assertEqual(plan["steps"], [])

    def test_invalid_reply_gets_format_feedback_on_retry(self):
        complete = json.dumps({
            "steps": [],
            "goal_reached": False,
            "goal_verification": "No target was supplied.",
        })
        with patch.object(planner, "call_model", side_effect=["No target supplied.", complete]) as model, patch.object(
            planner, "log_context_event"
        ):
            result = planner.generate_plan("Check a target")
        self.assertEqual(result["steps"], [])
        self.assertEqual(model.call_count, 2)
        self.assertIn("previous response could not be parsed", model.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
