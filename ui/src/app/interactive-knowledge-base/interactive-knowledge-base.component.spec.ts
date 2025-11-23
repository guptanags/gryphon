import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InteractiveKnowledgeBaseComponent } from './interactive-knowledge-base.component';

describe('InteractiveKnowledgeBaseComponent', () => {
  let component: InteractiveKnowledgeBaseComponent;
  let fixture: ComponentFixture<InteractiveKnowledgeBaseComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ InteractiveKnowledgeBaseComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(InteractiveKnowledgeBaseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
