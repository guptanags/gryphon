import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.service';
import { RepositoryStatus, IngestRequest } from '../api';

@Component({
  selector: 'app-insight-engine',
  templateUrl: './insight-engine.component.html',
  styleUrls: ['./insight-engine.component.scss']
})
export class InsightEngineComponent implements OnInit {
  repositories: RepositoryStatus[] = [];
  logicalName: string = '';
  gitRepo: string = '';
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

  ingest() {
    if (!this.logicalName || !this.gitRepo) {
      alert('Please provide a logical name and a git repo URL with branch.');
      return;
    }

    const request: IngestRequest = {
      logical_name: this.logicalName,
      git_repos: [this.gitRepo],
      confluence_pages: []
    };

    this.loading = true;
    this.dataService.ingest(request).subscribe(response => {
      console.log(response);
      this.loading = false;
      // Refresh the list of repositories after a short delay
      setTimeout(() => this.refreshRepositories(), 2000);
    });
  }
}