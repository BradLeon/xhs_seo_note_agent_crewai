"""Multi-agent crew for Xiaohongshu SEO optimization.

This module implements the CrewAI-based multi-agent system for analyzing
competitor notes and generating optimization strategies.
"""

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from typing import List, Dict, Any

from xhs_seo_optimizer.models.note import Note
from xhs_seo_optimizer.models.reports import (
    SuccessProfileReport,
    AuditReport,
    GapReport,
    OptimizationPlan,
)
from xhs_seo_optimizer.tools import (
    DataAggregatorTool,
    MultiModalVisionTool,
    NLPAnalysisTool,
    StatisticalDeltaTool,
    CompetitorAnalysisOrchestrator,
)


@CrewBase
class XhsSeoOptimizerCrew:
    """小红书SEO优化多智能体系统 (Xiaohongshu SEO Optimizer Multi-Agent System).

    This crew orchestrates 5 agents to analyze competitor notes and generate
    optimization strategies:
    1. Orchestrator - Manager agent (coordinates workflow in hierarchical process)
    2. CompetitorAnalyst - Analyzes why target_notes score high
    3. OwnedNoteAuditor - Audits owned_note characteristics (placeholder)
    4. GapFinder - Identifies gaps between owned and target notes (placeholder)
    5. OptimizationStrategist - Generates optimization plan (placeholder)

    The crew uses a hierarchical process where the Orchestrator manages task execution.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # === Callback Hooks ===

    @before_kickoff
    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs before crew execution.

        Args:
            inputs: Input dictionary with target_notes, keyword, etc.

        Returns:
            Validated inputs dictionary

        Raises:
            ValueError: If required inputs are missing
        """
        if not inputs.get("target_notes"):
            raise ValueError("target_notes is required")
        if not isinstance(inputs.get("target_notes"), list):
            raise ValueError("target_notes must be a list")
        if len(inputs.get("target_notes", [])) == 0:
            raise ValueError("At least one target note is required")

        if not inputs.get("keyword"):
            raise ValueError("keyword is required")

        # Mark inputs as validated
        inputs["validated"] = True
        return inputs

    @after_kickoff
    def process_output(self, output: Any) -> Any:
        """Process output after crew completion.

        Args:
            output: Crew execution output

        Returns:
            Processed output

        This hook can be used for:
        - Saving results to database
        - Sending notifications
        - Logging metrics
        """
        # For now, just log completion
        # In future, could save to database or send notifications
        print(f"Crew execution completed successfully")
        return output

    # === Agent Definitions ===

    @agent
    def orchestrator(self) -> Agent:
        """项目总监 agent (manager for hierarchical process).

        Returns:
            Agent instance configured as manager for hierarchical process
        """
        return Agent(
            config=self.agents_config['orchestrator'],
            tools=[]
        )

    @agent
    def competitor_analyst(self) -> Agent:
        """竞品分析师 agent.

        Analyzes target_notes to identify success patterns with statistical evidence.
        Uses CompetitorAnalysisOrchestrator for end-to-end analysis, plus individual tools
        for granular operations.

        Returns:
            Agent instance with tools configured
        """
        return Agent(
            config=self.agents_config['competitor_analyst'],
            tools=[
                CompetitorAnalysisOrchestrator(),  # Main orchestration tool
                DataAggregatorTool(),
                MultiModalVisionTool(),
                NLPAnalysisTool()
            ]
        )

    @agent
    def owned_note_auditor(self) -> Agent:
        """客户笔记诊断师 agent (placeholder).

        Placeholder for future implementation in change 0004.
        Audits owned_note characteristics and identifies strengths/weaknesses.

        Returns:
            Agent instance with tools configured
        """
        return Agent(
            config=self.agents_config['owned_note_auditor'],
            tools=[
                MultiModalVisionTool(),
                NLPAnalysisTool()
            ]
        )

    @agent
    def gap_finder(self) -> Agent:
        """差距定位员 agent (placeholder).

        Placeholder for future implementation in change 0005.
        Identifies gaps between owned_note and target_notes success patterns.

        Returns:
            Agent instance with tools configured
        """
        return Agent(
            config=self.agents_config['gap_finder'],
            tools=[StatisticalDeltaTool()]
        )

    @agent
    def optimization_strategist(self) -> Agent:
        """优化策略师 agent (placeholder).

        Placeholder for future implementation in change 0006.
        Generates concrete optimization recommendations based on gap analysis.

        Returns:
            Agent instance configured
        """
        return Agent(
            config=self.agents_config['optimization_strategist'],
            tools=[]
        )

    # === Task Definitions ===

    @task
    def orchestrate_workflow_task(self) -> Task:
        """协调工作流任务 (orchestrate workflow task).

        Placeholder for future implementation.
        Coordinates the entire analysis workflow.

        Returns:
            Task instance configured
        """
        return Task(
            config=self.tasks_config['orchestrate_workflow'],
            agent=self.orchestrator()
        )

    @task
    def analyze_competitors_task(self) -> Task:
        """竞品分析任务 (analyze competitors task).

        Analyzes target_notes to identify success patterns.
        Outputs SuccessProfileReport with feature patterns and creation formulas.

        Returns:
            Task instance configured with output_pydantic=SuccessProfileReport
        """
        return Task(
            config=self.tasks_config['analyze_competitors'],
            agent=self.competitor_analyst(),
            output_pydantic=SuccessProfileReport
        )

    @task
    def audit_owned_note_task(self) -> Task:
        """客户笔记审计任务 (audit owned note task).

        Placeholder for future implementation in change 0004.
        Audits owned_note characteristics.

        Returns:
            Task instance configured
        """
        return Task(
            config=self.tasks_config['audit_owned_note'],
            agent=self.owned_note_auditor(),
            output_pydantic=AuditReport
        )

    @task
    def find_gaps_task(self) -> Task:
        """差距定位任务 (find gaps task).

        Placeholder for future implementation in change 0005.
        Identifies gaps between owned_note and target_notes.

        Returns:
            Task instance configured with context dependencies
        """
        return Task(
            config=self.tasks_config['find_gaps'],
            agent=self.gap_finder(),
            output_pydantic=GapReport,
            # Context dependencies: this task needs outputs from analyze_competitors and audit_owned_note
            context=[
                self.analyze_competitors_task(),
                self.audit_owned_note_task()
            ]
        )

    @task
    def generate_strategy_task(self) -> Task:
        """生成优化策略任务 (generate strategy task).

        Placeholder for future implementation in change 0006.
        Generates optimization recommendations.

        Returns:
            Task instance configured with context dependencies
        """
        return Task(
            config=self.tasks_config['generate_strategy'],
            agent=self.optimization_strategist(),
            output_pydantic=OptimizationPlan,
            # Context dependency: this task needs output from find_gaps
            context=[self.find_gaps_task()]
        )

    # === Crew Assembly ===

    @crew
    def crew(self) -> Crew:
        """组装完整的crew.

        Returns:
            Crew instance with hierarchical process and Orchestrator as manager

        Note:
            Uses Process.hierarchical where Orchestrator manages task execution.
            For Phase 1 implementation, only analyze_competitors_task is fully functional.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.orchestrator(),
            verbose=True
        )
