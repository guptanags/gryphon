// src/app/settings/settings.component.ts
import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.service';

@Component({
  selector: 'app-settings',
  templateUrl: './settings.component.html',
  styleUrls: ['./settings.component.scss']
})
export class SettingsComponent implements OnInit {
  settingsData: any;

  constructor(private dataService: DataService) {}

  ngOnInit() {
    this.settingsData = this.dataService.getSettingsData();
  }

  updatePreference(pref: any) {
    console.log(`Updated ${pref.name} to ${pref.enabled}`);
    // Add logic to save preference (e.g., API call)
  }
}