import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from 'src/services/auth.service';


@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss']
})
export class RegisterComponent implements OnInit {
 
  show:boolean=false;
  nuser!:FormGroup
    
    constructor(private fb:FormBuilder, private router: Router,private auth:AuthService) { }
  
    ngOnInit(): void {
      this.nuser=this.fb.group(
        {
          username:['',Validators.required],
          pwd:['',Validators.required],
          name:['',Validators.required],
          mail:['',Validators.required]
  
  
        }
      
    )
      }
  
  
  onSubmit(){
    if(this.nuser.valid){
      this.auth.register(this.nuser.value.name,this.nuser.value.mail,this.nuser.value.username,this.nuser.value.pwd).subscribe({
        next:data=>{
          this.auth.storeToken(data.idToken);
          console.log("Registration successful");
          this.auth.canAutheticated()
          
        },
        error:data=>{
          if(data.error.error.message=="INVALID_EMAIL"){
            alert("Invalid email")
            this.auth.canAccess()
          }
          else if(data.error.error.message=="EMAIL_EXISTS"){
            alert("Email already exists")
            this.auth.canAccess()
          }
          else{
            alert("Unknown error occured while creating this account")
            this.auth.canAccess()
          }
        }
      })
      
    }

    
    else{
      this.validateAllFormFIelds(this.nuser)
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

