import { Injectable } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import * as XLSX from 'xlsx';
@Injectable({
  providedIn: 'root'
})
export class ResultService {
server_add="http://localhost:5000/dashboard"
  constructor(private http:HttpClient) { }
  send_post(df:any,range:number):Observable<any>{
    return this.http.post(this.server_add,[df,range])
  }
}
