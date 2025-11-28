import { Component, Input, OnInit, OnDestroy, SimpleChanges, OnChanges } from '@angular/core';
import { DataService } from '../data.service';
import { Subscription, interval } from 'rxjs';

interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
  width?: number;
  height?: number;
}

interface GraphEdge {
  from: string;
  to: string;
  conditional?: boolean;
}

@Component({
  selector: 'app-agent-graph',
  templateUrl: './agent-graph.component.html',
  styleUrls: ['./agent-graph.component.scss']
})
export class AgentGraphComponent implements OnInit, OnDestroy, OnChanges {
  @Input() logicalName: string | null = null;

  // Static graph definition mirroring `agent_graph.py`
  nodes: GraphNode[] = [
    { id: 'planner', label: 'Planner', x: 40, y: 20 },
    { id: 'researcher', label: 'Researcher', x: 260, y: 20 },
    { id: 'coder', label: 'Coder', x: 480, y: 20 },
    { id: 'syntax_checker', label: 'Syntax', x: 480, y: 140 },
    { id: 'reviewer', label: 'Reviewer', x: 260, y: 140 },
    { id: 'tester', label: 'Tester', x: 480, y: 260 },
    { id: 'git_manager', label: 'Git Manager', x: 260, y: 260 }
  ];

  edges: GraphEdge[] = [
    { from: 'planner', to: 'researcher' },
    { from: 'researcher', to: 'coder' },
    { from: 'coder', to: 'syntax_checker' },
    { from: 'syntax_checker', to: 'coder', conditional: true },
    { from: 'syntax_checker', to: 'reviewer' },
    { from: 'reviewer', to: 'coder', conditional: true },
    { from: 'reviewer', to: 'tester' },
    { from: 'tester', to: 'git_manager' }
  ];

  // Dynamic status
  currentStep: string | null = null;
  logs: string[] = [];
  status: string = 'idle';

  private pollSub: Subscription | null = null;

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    // Start polling if logicalName is present
    this.startPollingIfNeeded();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['logicalName']) {
      // When logicalName changes, reset polling and start again
      this.stopPolling();
      this.currentStep = null;
      this.logs = [];
      this.status = 'idle';
      this.startPollingIfNeeded();
    }
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  startPollingIfNeeded() {
    if (!this.logicalName) return;

    // Poll every 2 seconds
    this.pollSub = interval(2000).subscribe(() => {
      this.dataService.getAgentStatus(this.logicalName!).subscribe(res => {
        this.status = res.status;
        this.currentStep = res.current_step;
        this.logs = res.logs || [];
      }, err => {
        // It's ok to be noisy, just keep current state
        console.error('Failed to fetch agent status', err);
      });
    });
  }

  stopPolling() {
    if (this.pollSub) {
      this.pollSub.unsubscribe();
      this.pollSub = null;
    }
  }

  isCurrent(nodeId: string) {
    return nodeId === this.currentStep;
  }

  // helper to compute path between two nodes (straight line)
  getEdgePath(fromId: string, toId: string) {
    const from = this.nodes.find(n => n.id === fromId)!;
    const to = this.nodes.find(n => n.id === toId)!;
    const startX = from.x + (from.width ? from.width : 120);
    const startY = from.y + (from.height ? from.height : 40) / 2;
    const endX = to.x;
    const endY = to.y + (to.height ? to.height : 40) / 2;

    const midX = (startX + endX) / 2;
    return `M ${startX} ${startY} C ${midX} ${startY} ${midX} ${endY} ${endX} ${endY}`;
  }
}
