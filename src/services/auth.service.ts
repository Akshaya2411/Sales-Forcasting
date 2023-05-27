import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
@Injectable({
  providedIn: 'root'
})
export class AuthService {

  constructor(private router:Router,private http:HttpClient) { }
  isAuthenticated():boolean{
    if(sessionStorage.getItem('token')!==null){
      return true;
    }
    return false;
  }
canAccess(){
if(!this.isAuthenticated()){
  this.router.navigate(['/home'])
}
}
canAutheticated(){
  if(this.isAuthenticated()){
    this.router.navigate(['/dashboard'])
  }
  }

register(name:string,mail:string,username:string,pwd:string){
  return this.http.post<{idToken:string}>("https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyCLMxXAK_09SIbaMUVFiNW-T1tvt8avCh0",
  {displayName:name,email:mail,username:username,password:pwd});
}
login(mail:string,pwd:string){
  return this.http.post<{idToken:string}>("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyCLMxXAK_09SIbaMUVFiNW-T1tvt8avCh0",
  {email:mail,password:pwd});
}

storeToken(token:string){
  sessionStorage.setItem('token',token)
}

removeToken(){
  sessionStorage.removeItem('token');
  this.router.navigate(['/'])
}
}