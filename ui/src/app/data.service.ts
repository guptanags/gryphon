import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  IngestRequest,
  IngestResponse,
  RepositoryList,
  TestGenRequest,
  TestGenResponse,
  QueryRequest,
  QueryResponse,
  MetricsUpdate,
  AutonomousTaskRequest,
  AgentStatusResponse
} from './api';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiUrl = 'http://localhost:8000'; // Backend API URL

  constructor(private http: HttpClient) {}

  ingest(request: IngestRequest): Observable<IngestResponse> {
    return this.http.post<IngestResponse>(`${this.apiUrl}/ingest`, request);
  }

  getRepositories(): Observable<RepositoryList> {
    return this.http.get<RepositoryList>(`${this.apiUrl}/repositories`);
  }

  generateTests(request: TestGenRequest): Observable<TestGenResponse> {
    return this.http.post<TestGenResponse>(`${this.apiUrl}/generate-tests`, request);
  }

  query(request: QueryRequest): Observable<QueryResponse> {
    return this.http.post<QueryResponse>(`${this.apiUrl}/query`, request);
  }

  updateCoverageMetrics(request: MetricsUpdate): Observable<any> {
    return this.http.post(`${this.apiUrl}/metrics/coverage`, request);
  }

  runAgent(request: AutonomousTaskRequest): Observable<any> {
    return this.http.post(`${this.apiUrl}/agent/run`, request);
  }

  getAgentStatus(logicalName: string): Observable<AgentStatusResponse> {
    return this.http.get<AgentStatusResponse>(`${this.apiUrl}/agent/status/${logicalName}`);
  }

  getSettingsData() {
    return {
      preferences: [
        { name: 'Dark Theme', enabled: true },
        { name: 'Auto-Save', enabled: false },
        { name: 'Notifications', enabled: true }
      ],
      user: {
        name: 'User',
        email: 'user@example.com'
      }
    };
  }
}