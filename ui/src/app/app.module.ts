// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { InsightEngineComponent } from './insight-engine/insight-engine.component';
import { QualityGuardianComponent } from './quality-guardian/quality-guardian.component';
import { InteractiveKnowledgeBaseComponent } from './interactive-knowledge-base/interactive-knowledge-base.component';
import { AutonomousCoPilotComponent } from './autonomous-co-pilot/autonomous-co-pilot.component';
import { SettingsComponent } from './settings/settings.component';
import { CodeQualityGuardianComponent } from './code-quality-guardian/code-quality-guardian.component';
import { RouterModule, Routes } from '@angular/router';
import { SidebarComponent } from './sidebar/sidebar.component';
import { FormsModule } from '@angular/forms'; // For ngModel
import { HttpClientModule } from '@angular/common/http';

const routes: Routes = [
  { path: 'dashboard', component: DashboardComponent },
  { path: 'insight-engine', component: InsightEngineComponent },
  { path: 'quality-guardian', component: QualityGuardianComponent },
  { path: 'interactive-knowledge-base', component: InteractiveKnowledgeBaseComponent },
  { path: 'autonomous-co-pilot', component: AutonomousCoPilotComponent },
  { path: 'code-quality-guardian', component: CodeQualityGuardianComponent },
  { path: 'settings', component: SettingsComponent },
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' } // Set dashboard as home
];

@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    InsightEngineComponent,
    QualityGuardianComponent,
    InteractiveKnowledgeBaseComponent,
    AutonomousCoPilotComponent,
    SettingsComponent,
    CodeQualityGuardianComponent,
    SidebarComponent
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(routes),
    FormsModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}