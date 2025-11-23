// This file defines the interfaces for the API communication between the Angular UI and the FastAPI backend.

export interface IngestRequest {
  logical_name: string;
  git_repos: string[];
  confluence_pages: string[];
}

export interface IngestResponse {
  message: string;
  logical_name: string;
  status_url: string;
}

export interface RepositoryStatus {
  logical_name: string;
  repo_url: string;
  branch_name: string;
  status: string;
  test_status: string;
  last_ingested_at: string | null;
  code_coverage_score: number | null;
  code_quality_score: number | null;
  code_static_analysis_score: number | null;
}

export interface RepositoryList {
  repositories: RepositoryStatus[];
}

export interface QueryRequest {
  question: string;
  top_k?: number;
}

export interface QueryResponse {
  answer: string;
  code_context: string;
  doc_context: string;
}

export interface TestGenRequest {
  logical_name: string;
  test_types: string[];
}

export interface TestGenResponse {
  message: string;
  logical_name: string;
  status_url: string;
}

export interface MetricsUpdate {
  repo_url: string;
  branch_name: string;
  coverage_score: number;
}

export interface AutonomousTaskRequest {
  logical_name: string;
  requirement: string;
}

export interface AgentStatusResponse {
  logical_name: string;
  status: string;
  current_step: string;
  logs: string[];
}

