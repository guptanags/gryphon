import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { DataService } from '../data.service';
import { RepositoryStatus, TestGenRequest, AutonomousTaskRequest } from '../api';

@Component({
  selector: 'app-repository-detail',
  templateUrl: './repository-detail.component.html',
  styleUrls: ['./repository-detail.component.scss']
})
export class RepositoryDetailComponent implements OnInit {
  logicalName: string = '';
  repository: RepositoryStatus | null = null;
  loading: boolean = false;
  testGenerating: boolean = false;
  agentRunning: boolean = false;
  agentStatus: any = null;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private dataService: DataService
  ) {}

  ngOnInit(): void {
    this.logicalName = this.route.snapshot.paramMap.get('logicalName') || '';
    if (!this.logicalName) {
      // if no logical name, go back to dashboard
      this.router.navigate(['/dashboard']);
      return;
    }
    this.loadRepository();
  }

  loadRepository() {
    this.loading = true;
    this.dataService.getRepositories().subscribe(data => {
      const found = data.repositories.find(r => r.logical_name === this.logicalName);
      this.repository = found || null;
      this.loading = false;
    }, err => {
      console.error(err);
      this.loading = false;
    });
  }

  // Gauge helpers (same as dashboard)
  getGaugeArcPath(value: number, max: number): string {
    const percentage = Math.min(Math.max(value / max, 0), 1);
    const angle = percentage * 140;
    const radian = (angle - 90) * Math.PI / 180;
    const radius = 70;
    const centerX = 100;
    const centerY = 100;
    const endX = centerX + radius * Math.cos(radian);
    const endY = centerY + radius * Math.sin(radian);
    const largeArc = angle > 70 ? 1 : 0;
    return `M 30 100 A 70 70 0 ${largeArc} 1 ${endX} ${endY}`;
  }

  getGaugeColor(percentage: number): string {
    if (percentage < 30) return '#f44336';
    if (percentage < 60) return '#ff9800';
    if (percentage < 80) return '#ffc107';
    return '#4caf50';
  }

  generateTests() {
    if (!this.repository) return;
    this.testGenerating = true;
    const req: TestGenRequest = { logical_name: this.repository.logical_name, test_types: ['unit'] };
    this.dataService.generateTests(req).subscribe(res => {
      this.testGenerating = false;
      alert('Test generation queued. Check Repositories for status.');
    }, err => {
      console.error(err);
      this.testGenerating = false;
      alert('Failed to queue test generation');
    });
  }

  runCodeReview() {
    if (!this.repository) return;
    this.agentRunning = true;
    const req: AutonomousTaskRequest = { logical_name: this.repository.logical_name, requirement: 'Perform code review and suggest optimizations' };
    this.dataService.runAgent(req).subscribe(res => {
      this.agentRunning = false;
      alert('Agent started for code review. Use Agent Status to monitor progress.');
    }, err => {
      console.error(err);
      this.agentRunning = false;
      alert('Failed to start agent.');
    });
  }

  refresh() {
    this.loadRepository();
  }
}
