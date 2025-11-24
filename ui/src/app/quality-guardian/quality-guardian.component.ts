import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.service';
import { RepositoryStatus, TestGenRequest } from '../api';

@Component({
  selector: 'app-quality-guardian',
  templateUrl: './quality-guardian.component.html',
  styleUrls: ['./quality-guardian.component.scss']
})
export class QualityGuardianComponent implements OnInit {
  repositories: RepositoryStatus[] = [];
  selectedLogicalName: string = '';
  testTypes = ['unit', 'acceptance', 'load'];
  selectedTestTypes: { [key: string]: boolean } = {};
  loading = false;
  displayedColumns: string[] = ['logical_name', 'status', 'test_status'];

  constructor(private dataService: DataService) {}

  ngOnInit() {
    this.refreshRepositories();
    this.testTypes.forEach(t => this.selectedTestTypes[t] = false);
  }

  refreshRepositories() {
    this.loading = true;
    this.dataService.getRepositories().subscribe(data => {
      this.repositories = data.repositories;
      this.loading = false;
    });
  }

  generateAndExecute() {
    const selectedTypes = Object.keys(this.selectedTestTypes).filter(t => this.selectedTestTypes[t]);

    if (!this.selectedLogicalName || selectedTypes.length === 0) {
      alert('Please select a repository and at least one test type.');
      return;
    }

    const request: TestGenRequest = {
      logical_name: this.selectedLogicalName,
      test_types: selectedTypes
    };

    this.loading = true;
    this.dataService.generateTests(request).subscribe(response => {
      console.log(response);
      this.loading = false;
      // Refresh the list of repositories after a short delay to show test status
      setTimeout(() => this.refreshRepositories(), 2000);
    });
  }
}