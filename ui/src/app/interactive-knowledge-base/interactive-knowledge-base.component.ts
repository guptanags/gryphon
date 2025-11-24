import { Component } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { DataService } from '../data.service';
import { QueryRequest, QueryResponse } from '../api';
import { marked } from 'marked';

@Component({
  selector: 'app-interactive-knowledge-base',
  templateUrl: './interactive-knowledge-base.component.html',
  styleUrls: ['./interactive-knowledge-base.component.scss']
})
export class InteractiveKnowledgeBaseComponent {
  question: string = '';
  response: QueryResponse | null = null;
  loading: boolean = false;
  answerHtml: SafeHtml = '';
  codeContextHtml: SafeHtml = '';
  docContextHtml: SafeHtml = '';

  constructor(private dataService: DataService, private sanitizer: DomSanitizer) {}

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
      async (data) => {
        this.response = data;
        // Convert markdown to HTML
        this.answerHtml = this.sanitizer.bypassSecurityTrustHtml(await marked(data.answer));
        this.codeContextHtml = this.sanitizer.bypassSecurityTrustHtml(await marked(data.code_context));
        this.docContextHtml = this.sanitizer.bypassSecurityTrustHtml(await marked(data.doc_context));
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