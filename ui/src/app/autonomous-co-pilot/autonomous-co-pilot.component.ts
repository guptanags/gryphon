import { Component, OnInit, OnDestroy } from '@angular/core';
import { DataService } from '../data.service';
import { RepositoryStatus, AgentStatusResponse, AutonomousTaskRequest } from '../api';

@Component({
  selector: 'app-autonomous-co-pilot',
  templateUrl: './autonomous-co-pilot.component.html',
  styleUrls: ['./autonomous-co-pilot.component.scss']
})
export class AutonomousCoPilotComponent implements OnInit, OnDestroy {
  repositories: RepositoryStatus[] = [];
  selectedLogicalName: string = '';
  requirement: string = '';
  agentStatus: AgentStatusResponse | null = null;
  loading: boolean = false;
  pollingInterval: any;

  constructor(private dataService: DataService) {}

  ngOnInit() {
    this.dataService.getRepositories().subscribe(data => {
      this.repositories = data.repositories;
    });
  }

  runAgent() {
    if (!this.selectedLogicalName || !this.requirement) {
      alert('Please select a repository and provide a requirement.');
      return;
    }

    const request: AutonomousTaskRequest = {
      logical_name: this.selectedLogicalName,
      requirement: this.requirement
    };

    this.loading = true;
    this.agentStatus = null;
    this.dataService.runAgent(request).subscribe(() => {
      this.startPolling();
    });
  }

  startPolling() {
    this.pollingInterval = setInterval(() => {
      this.pollStatus();
    }, 5000); // Poll every 5 seconds
  }

  pollStatus() {
    if (!this.selectedLogicalName) {
      return;
    }

    this.dataService.getAgentStatus(this.selectedLogicalName).subscribe(status => {
      this.agentStatus = status;
      if (status.status === 'completed' || status.status === 'failed') {
        this.stopPolling();
        this.loading = false;
      }
    });
  }

  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }

  ngOnDestroy() {
    this.stopPolling();
  }
}