import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.service';
import { RepositoryStatus } from '../api';

@Component({
  selector: 'app-code-quality-guardian',
  templateUrl: './code-quality-guardian.component.html',
  styleUrls: ['./code-quality-guardian.component.scss']
})
export class CodeQualityGuardianComponent implements OnInit {
  repositories: RepositoryStatus[] = [];
  loading: boolean = false;
  displayedColumns: string[] = ['logical_name', 'code_quality_score', 'code_static_analysis_score', 'code_coverage_score'];

  constructor(private dataService: DataService) {}

  ngOnInit() {
    this.refreshRepositories();
  }

  refreshRepositories() {
    this.loading = true;
    this.dataService.getRepositories().subscribe(data => {
      this.repositories = data.repositories;
      this.loading = false;
    });
  }
}