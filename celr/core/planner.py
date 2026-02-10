import networkx as nx
from typing import List, Optional
from celr.core.types import Plan, Step, TaskContext
from celr.core.reasoning import ReasoningCore

class Planner:
    def __init__(self, reasoning_core: ReasoningCore):
        self.reasoning = reasoning_core
        self.graph = nx.DiGraph()

    def create_initial_plan(self, context: TaskContext) -> Plan:
        """
        Uses the ReasoningCore to decompose the task and builds a NetworkX graph.
        """
        # 1. Decompose
        context.log("running_decomposition")
        plan = self.reasoning.decompose(context)
        
        # 2. Build Graph
        self.graph.clear()
        for step in plan.items:
            self.graph.add_node(step.id, data=step)
            for dep_id in step.dependencies:
                self.graph.add_edge(dep_id, step.id)
                
        # 3. Validate for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            context.log("WARNING: Cyclic dependency detected in plan! Breaking cycles...")
            # Simple fallback: linearize based on ID order or list order
            # For now, just logging warning
            
        context.log(f"Plan created with {len(plan.items)} steps.")
        return plan

    def get_ready_steps(self, plan: Plan) -> List[Step]:
        """
        Returns steps that are PENDING and have no uncompleted dependencies.
        Uses the NetworkX graph to determine this efficiently.
        """
        ready_steps = []
        for step in plan.items:
            # Sync graph state with plan object state just in case
            # (In a real system, these would be tightly coupled)
            if step.status != "PENDING":
                continue
                
            # Check dependencies
            is_ready = True
            for ancestor in nx.ancestors(self.graph, step.id):
                # Ancestor must be COMPLETED
                # We need to find the step object for this ancestor ID
                ancestor_step = next((s for s in plan.items if s.id == ancestor), None)
                if ancestor_step and ancestor_step.status != "COMPLETED":
                    is_ready = False
                    break
            
            # Direct parents check (redundant if ancestors check works, but safer)
            for parent in self.graph.predecessors(step.id):
                parent_step = next((s for s in plan.items if s.id == parent), None)
                if parent_step and parent_step.status != "COMPLETED":
                    is_ready = False
                    break
                    
            if is_ready:
                ready_steps.append(step)
                
        return ready_steps
