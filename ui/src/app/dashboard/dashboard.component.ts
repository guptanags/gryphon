import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.service';
import { Router } from '@angular/router';
import { RepositoryStatus } from '../api';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  repositories: RepositoryStatus[] = [];
  loading: boolean = false;
  totalRepositories: number = 0;
  averageCoverage: number = 0;
  averageQualityScore: number = 0;
  displayedColumns: string[] = ['logical_name', 'status', 'test_status'];

  constructor(private dataService: DataService, private router: Router) {}

  openRepository(logicalName: string) {
    this.router.navigate(['/repository', logicalName]);
  }

  ngOnInit() {
    this.refresh();
  }

  refresh() {
    this.loading = true;
    this.dataService.getRepositories().subscribe(data => {
      this.repositories = data.repositories;
      this.calculateMetrics();
      this.loading = false;
    });
  }

  calculateMetrics() {
    this.totalRepositories = this.repositories.length;

    if (this.totalRepositories === 0) {
      this.averageCoverage = 0;
      this.averageQualityScore = 0;
      return;
    }

    const totalCoverage = this.repositories
      .map(r => r.code_coverage_score || 0)
      .reduce((sum, current) => sum + current, 0);
    this.averageCoverage = totalCoverage / this.repositories.filter(r => r.code_coverage_score !== null).length || 0;

    const totalQualityScore = this.repositories
      .map(r => r.code_quality_score || 0)
      .reduce((sum, current) => sum + current, 0);
    this.averageQualityScore = totalQualityScore / this.repositories.filter(r => r.code_quality_score !== null).length || 0;
  }

  getGaugeArcPath(value: number, max: number): string {
    // Clamp value between 0 and max
    const percentage = Math.min(Math.max(value / max, 0), 1);
    
    // Arc goes from 30 to 170 on x-axis (140 degrees)
    // Convert percentage to angle (0-140 degrees)
    const angle = percentage * 140;
    const radian = (angle - 90) * Math.PI / 180;
    
    // Calculate end point
    const radius = 70;
    const centerX = 100;
    const centerY = 100;
    const endX = centerX + radius * Math.cos(radian);
    const endY = centerY + radius * Math.sin(radian);
    
    // Determine if we need large arc flag
    const largeArc = angle > 70 ? 1 : 0;
    
    return `M 30 100 A 70 70 0 ${largeArc} 1 ${endX} ${endY}`;
  }

  getGaugeColor(percentage: number): string {
    // Red for < 30, Orange for < 60, Yellow for < 80, Green for >= 80
    if (percentage < 30) return '#f44336'; // Red
    if (percentage < 60) return '#ff9800'; // Orange
    if (percentage < 80) return '#ffc107'; // Yellow
    return '#4caf50'; // Green
  }
}