
# Gryphon AI

## Overview
Gryphon AI is a modular developer productivity platform built with Angular 15. It provides:
- **Dashboard**: Centralized view of all modules and project status.
- **Insight Engine**: Ingests repositories, allows users to select documentation types, and generates documentation for codebases.
- **Quality Guardian**: Accepts a repository, lets users generate and execute various test cases (unit, acceptance, automation, load), and displays results.
- **Autonomous Co-pilot**: Accepts feature requests, performs impact analysis, generates code, and automates pull requests.
- **Interactive Knowledge Base**: Provides a searchable, interactive knowledge base for your code and documentation.
- **Intelli Review**: Automated code review and quality checks.
- **Settings**: Manage user and app preferences.

## Functional Specs
- Unified sidebar navigation for all modules.
- Each module is a self-contained Angular component.
- Insight Engine supports multi-URL input and documentation type selection.
- Quality Guardian supports repository input, test type selection, and displays test results and suite details.
- Autonomous Co-pilot supports feature request input, workflow steps, and PR automation.
- All modules share a consistent color theme and logo branding.

## Technical Documentation

### Prerequisites
- **Node.js**: v16.x or v18.x recommended
- **npm**: v8.x or later
- **Angular CLI**: v15.x

### Setup Instructions
1. **Clone the repository:**
	```sh
	git clone <your-repo-url>
	cd gryphon-ai
	```
2. **Install Node.js and npm:**
	- Download from [nodejs.org](https://nodejs.org/)
	- Verify installation:
	  ```sh
	  node -v
	  npm -v
	  ```
3. **Install Angular CLI 15:**
	```sh
	npm install -g @angular/cli@15
	```
	- Verify:
	  ```sh
	  ng version
	  ```
4. **Install dependencies:**
	```sh
	npm install
	```
5. **Run the development server:**
	```sh
	ng serve
	```
	- Open [http://localhost:4200](http://localhost:4200) in your browser.

### Build
```sh
ng build
```
Build artifacts are stored in the `dist/` directory.

### Testing
- **Unit tests:**
  ```sh
  ng test
  ```
- **End-to-end tests:**
  ```sh
  ng e2e
  ```

### Code Generation & Scaffolding
Generate a new component:
```sh
ng generate component component-name
```
Other schematics: `ng generate directive|pipe|service|class|guard|interface|enum|module`

### Further Help
For more help on Angular CLI, use `ng help` or see the [Angular CLI Documentation](https://angular.io/cli).
