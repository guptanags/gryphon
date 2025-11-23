import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QualityGuardianComponent } from './quality-guardian.component';

describe('QualityGuardianComponent', () => {
  let component: QualityGuardianComponent;
  let fixture: ComponentFixture<QualityGuardianComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ QualityGuardianComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(QualityGuardianComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
