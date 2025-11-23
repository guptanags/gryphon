import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CodeQualityGuardianComponent } from './code-quality-guardian.component';

describe('CodeQualityGuardianComponent', () => {
  let component: CodeQualityGuardianComponent;
  let fixture: ComponentFixture<CodeQualityGuardianComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CodeQualityGuardianComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CodeQualityGuardianComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
