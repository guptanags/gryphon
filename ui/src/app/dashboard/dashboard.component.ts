import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.service';
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

  constructor(private dataService: DataService) {}

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
}