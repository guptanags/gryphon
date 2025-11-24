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
import { SidebarComponent } from './sidebar/sidebar.component';
import { RepositoryDetailComponent } from './repository-detail/repository-detail.component';
import { RouterModule, Routes } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { LayoutModule } from '@angular/cdk/layout';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatTableModule } from '@angular/material/table';
import { MatSelectModule } from '@angular/material/select';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';

const routes: Routes = [
  { path: 'dashboard', component: DashboardComponent },
  { path: 'insight-engine', component: InsightEngineComponent },
  { path: 'quality-guardian', component: QualityGuardianComponent },
  { path: 'interactive-knowledge-base', component: InteractiveKnowledgeBaseComponent },
  { path: 'autonomous-co-pilot', component: AutonomousCoPilotComponent },
  { path: 'code-quality-guardian', component: CodeQualityGuardianComponent },
  { path: 'repository/:logicalName', component: RepositoryDetailComponent },
  { path: 'settings', component: SettingsComponent },
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' }
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
    SidebarComponent,
    RepositoryDetailComponent
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(routes),
    FormsModule,
    HttpClientModule,
    BrowserAnimationsModule,
    LayoutModule,
    MatSidenavModule,
    MatToolbarModule,
    MatListModule,
    MatIconModule,
    MatButtonModule,
    MatCardModule,
    MatInputModule,
    MatFormFieldModule,
    MatTableModule,
    MatSelectModule,
    MatCheckboxModule,
    MatSlideToggleModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}