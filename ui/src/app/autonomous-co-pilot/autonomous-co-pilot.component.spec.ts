import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AutonomousCoPilotComponent } from './autonomous-co-pilot.component';

describe('AutonomousCoPilotComponent', () => {
  let component: AutonomousCoPilotComponent;
  let fixture: ComponentFixture<AutonomousCoPilotComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AutonomousCoPilotComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AutonomousCoPilotComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
