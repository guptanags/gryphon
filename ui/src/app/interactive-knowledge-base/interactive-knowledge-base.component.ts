import { Component } from '@angular/core';
import { DataService } from '../data.service';
import { QueryRequest, QueryResponse } from '../api';

@Component({
  selector: 'app-interactive-knowledge-base',
  templateUrl: './interactive-knowledge-base.component.html',
  styleUrls: ['./interactive-knowledge-base.component.scss']
})
export class InteractiveKnowledgeBaseComponent {
  question: string = '';
  response: QueryResponse | null = null;
  loading: boolean = false;

  constructor(private dataService: DataService) {}

  sendQuestion() {
    if (!this.question.trim()) {
      return;
    }

    const request: QueryRequest = {
      question: this.question
    };

    this.loading = true;
    this.response = null;
    this.dataService.query(request).subscribe(
      (data) => {
        this.response = data;
        this.loading = false;
      },
      (error) => {
        console.error(error);
        this.loading = false;
        alert('An error occurred while fetching the answer.');
      }
    );
  }
}