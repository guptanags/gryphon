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