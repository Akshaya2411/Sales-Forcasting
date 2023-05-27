import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from 'src/services/auth.service';
@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {
show:boolean=false;
user!:FormGroup
  
  constructor(private fb:FormBuilder, private router: Router,private auth:AuthService) { }

  ngOnInit(): void {
    this.user=this.fb.group(
      {
        mail:['',Validators.required],
        pwd:['',Validators.required]

      }
    )
  }

onSubmit(){
  if(this.user.valid){
    this.auth.login(this.user.value.mail,this.user.value.pwd).subscribe({
      next:data=>{
        this.auth.storeToken(data.idToken);
        console.log("Login successful");
        this.auth.canAutheticated();

      },
      error:data=>{
        if(data.error.error.message=="INVALID_EMAIL" || data.error.error.message=="INVALID_EMAIL"){
          alert("Invalid Credentials")
          this.auth.canAccess()
        }
     
        else{
          alert("Unknown error occured while loggin in")
          this.auth.canAccess()
        }
      }
    })
    
  }
  
  else{
    this.validateAllFormFIelds(this.user)
    alert("Your form is invalid")
  }
  }
  
  
  private validateAllFormFIelds(formGroup:FormGroup){
    Object.keys(formGroup.controls).forEach(field=>{
      const control=formGroup.get(field);
      if(control instanceof FormControl){
        control.markAsDirty({onlySelf:true});
      }
      else if(control instanceof FormGroup){
        this.validateAllFormFIelds(control)
      }
    })
  }
  
  password(){
    this.show=!this.show;
  }

}
