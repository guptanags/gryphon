// src/app/data.service.ts
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  // Existing methods from previous dashboards
  getCodeToDocsData() {
    return {
      codeSnippets: [
        `automatically document ( { foo: 'bar', algo: 'complex' } )`,
        `public int calculate(int a, int b) { return a + b; }`,
        `puggets ( { built: deploy: fast } )`,
        `const greet = (name) => console.log(\`Hello, \${name}!\`);`
      ]
    };
  }

  getQualityGuardianData() {
    return {
      passRate: 92,
      passed: 145,
      failed: 12,
      skipped: 3,
      recentRuns: [
        { date: '2024-03-08', status: 'Passed', count: 5 },
        { date: '2024-03-07', status: 'Failed', count: 5 },
        { date: '2024-03-07', status: 'Failed', count: 4 },
        { date: '2024-03-07', status: 'Failed', count: 4 }
      ],
      testSuiteDetails: [
        { name: 'Authentication_Module', tests: [
          { name: 'Test_Login_Success', status: 'Passed' },
          { name: 'Test_Login_InvalidCredentials', status: 'Failed' },
          { name: 'Test_Logout_Functionality', status: 'Passed' }
        ]}
      ],
      generatedTest: `test_Login_Success() {
        // Generated Test
        const { username, password } = { username: 'test', password: 'test' };
        expect(true).toBe(true);
      }`
    };
  }

  getInsightEngineData() {
    return {
      repositories: [
        { name: 'my-codebase', ingested: true }
      ],
      documentationTypes: [
        { name: 'Functional Documentation', selected: false },
        { name: 'Technical Documentation', selected: false }
      ]
    };
  }

  getKnowledgeBaseData() {
    return {
      ragQa: {
        question: 'AI reurcad tinter gue le w .ffts example mute n rem thoe, eted doueration t n ay example.',
        response: 'QA02QU9://https://socetoatioo'
      }
    };
  }

  getAutonomousCoPilotData() {
    return {
      features: [
        { name: 'Generate Boilerplate', enabled: true },
        { name: 'Refactor Function', enabled: false },
        { name: 'Suggest Reviewers', enabled: true }
      ]
    };
  }
}