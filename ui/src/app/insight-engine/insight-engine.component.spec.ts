import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InsightEngineComponent } from './insight-engine.component';

describe('InsightEngineComponent', () => {
  let component: InsightEngineComponent;
  let fixture: ComponentFixture<InsightEngineComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ InsightEngineComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(InsightEngineComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
