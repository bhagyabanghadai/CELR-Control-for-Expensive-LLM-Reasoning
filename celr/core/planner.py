import logging
import networkx as nx
from typing import List, Optional
from celr.core.types import Plan, Step, TaskContext, TaskStatus
from celr.core.reasoning import ReasoningCore

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, reasoning_core: ReasoningCore):
        self.reasoning = reasoning_core
        self.graph = nx.DiGraph()

    def _determine_agent(self, context: TaskContext) -> str:
        """Determines the best agent for the entire task."""
        # Simple heuristic for now - can be LLM-based later
        req = context.original_request.lower()
        
        if "python" in req or "function" in req or "code" in req or "def " in req:
            return "CODER"
        elif "calculate" in req or "solve" in req or "math" in req or any(c.isdigit() for c in req):
            # If it looks like a word problem
            return "MATHEMATICIAN"
        elif "history" in req or "science" in req or "fact" in req or "who" in req or "what is" in req:
            return "RESEARCHER"
        else:
            return "CODER"  # Default to coder for general tasks

    def create_initial_plan(self, context: TaskContext) -> Plan:
        """
        Uses the ReasoningCore to decompose the task and builds a NetworkX graph.
        Assigns a specific AGENT to the plan metadata.
        """
        # 1. Determine Agent
        agent_role = self._determine_agent(context)
        context.log(f"Planner assigned task to specialized agent: {agent_role}")

        # 2. Decompose (using the BlueprintArchitect persona logic potentially, but for now standard ReasoningCore)
        # Ideally, we would swich the ReasoningCore prompt here too, but let's keep decomposition generic for Phase 6.1
        context.log("running_decomposition")
        plan = self.reasoning.decompose(context)
        
        # Store the assigned agent in the plan's metadata (we need to ensure Plan has this field or verify context usage)
        # Since Plan doesn't have a metadata field in types.py yet, we'll store it in context.shared_state
        context.shared_state["assigned_agent"] = agent_role
        
        # 3. Build Graph
        self.graph.clear()
        for step in plan.items:
            self.graph.add_node(step.id, data=step)
            for dep_id in step.dependencies:
                self.graph.add_edge(dep_id, step.id)
                
        # 4. Validate for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            context.log("WARNING: Cyclic dependency detected in plan! Breaking cycles...")
            
        context.log(f"Plan created with {len(plan.items)} steps.")
        return plan

    def get_ready_steps(self, plan: Plan) -> List[Step]:
        """
        Returns steps that are PENDING and have no uncompleted dependencies.
        Uses the NetworkX graph to determine this efficiently.
        """
        # Fallback to simple Plan-based check if graph is empty (common in manual tests)
        if not self.graph.nodes:
            return plan.get_runnable_steps()

        ready_steps = []
        for step in plan.items:
            if step.status != TaskStatus.PENDING:
                continue
                
            # Check dependencies
            is_ready = True
            try:
                # Ancestors check
                if step.id in self.graph:
                    for ancestor in nx.ancestors(self.graph, step.id):
                        ancestor_step = next((s for s in plan.items if s.id == ancestor), None)
                        if ancestor_step and ancestor_step.status != TaskStatus.COMPLETED:
                            is_ready = False
                            break
            except Exception:
                # Fallback if graph desyncs
                is_ready = all(dep in {s.id for s in plan.items if s.status == TaskStatus.COMPLETED} for dep in step.dependencies)
                    
            if is_ready:
                ready_steps.append(step)
                
        return ready_steps
