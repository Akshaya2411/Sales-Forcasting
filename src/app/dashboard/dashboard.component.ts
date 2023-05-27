import { Component, OnInit } from '@angular/core';
import { AuthService } from 'src/services/auth.service';
import * as XLSX from 'xlsx';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ResultService } from 'src/services/result.service';
import { ngxCsv } from 'ngx-csv/ngx-csv';
import { keys } from 'lodash-es';
@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
dataset!:[][];
result_data:any
sales_pred:any
sales=[]
sales_file=[]
img_data:any
mape:any
mae:any
rmse:any
submit:boolean=false
hide_data:boolean=false;
img:Array<Object> =[];
title=['Components','Forecast','Actual Vs Predictetd']
f!:FormGroup
  constructor(private fb:FormBuilder,private auth:AuthService,private result:ResultService) { }

  ngOnInit(): void {
    this.auth.canAccess();
    this.f=this.fb.group(
      {
        file:['',Validators.required],
        target:['',Validators.required],
        period:['',Validators.required],
        range:['',Validators.required]

      }
    );
   
  }
  
  onFileChange(evt:any){
   const target:DataTransfer=<DataTransfer>(evt.target);
   if(target.files.length!==1) throw new Error("Only one file can be uploaded");
   const reader:FileReader=new FileReader();
   reader.onload=(e:any)=>{
   const bstr:string=e.target.result;
   const wb:XLSX.WorkBook=XLSX.read(bstr,{type:'binary'});
  const wsname:string=wb.SheetNames[0];
  const ws:XLSX.WorkSheet=wb.Sheets[wsname];
  console.log(ws);
  this.dataset=(XLSX.utils.sheet_to_json(ws,{header:1}));
  console.log(this.dataset)
   }
   reader.readAsBinaryString(target.files[0]);
  }
onsubmit(event:Event){
event.preventDefault()
this.submit=true
if(this.f.value.period=='week'){
  this.f.value.range=this.f.value.range*7
}
else if(this.f.value.period=='month'){
  this.f.value.range=this.f.value.range*30
}
else if(this.f.value.period=='year'){
  this.f.value.range=this.f.value.range*365
}
if(this.f.value.period=='day'){
  this.f.value.range=this.f.value.range
}
this.result.send_post(this.dataset,this.f.value.range).subscribe(data=>{
  console.log(data)
  this.result_data=data
  this.sales_pred=this.result_data[0]
  this.img_data=this.result_data[1]
  this.mape=this.result_data[2]
  this.mae=this.result_data[3]
  this.rmse=this.result_data[4]
  console.log(this.sales_pred)
  this.sales_pred=JSON.parse(this.sales_pred)
  for(let i in this.sales_pred){
     let sales_array=Array()
     for(let v in this.sales_pred[i]){
       sales_array.push(this.sales_pred[i][v])
     }
     this.sales.push(sales_array)
   }
  console.log(this.sales)
  console.log(typeof(this.sales))
  this.hide_data=true
  var x=0
  for(let i in this.img_data){
    i=this.img_data[i]
    this.img.push({
      image:'data:image/png;base64,'+i,
      thumbImage:'data:image/png;base64,'+i,
      title:this.title[x]
    })
    x=x+1
  }
})
this.f.reset()
this.hide_data=false
}
download(){
  let options = {
    fieldSeparator: ',',
    quoteStrings: '"',
    decimalseparator: '.',
    showLabels: true, 
    showTitle: false,
    useBom: true,
    noDownload: false,
    headers:['Date','yhat','Actual Sales','Predicted Sales']
    }
    new ngxCsv(this.sales,'predictions',options)
  
}

exit() {
  location.reload();
}
}
