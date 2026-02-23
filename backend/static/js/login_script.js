const container = document.getElementById('container');
const registerBtn = document.getElementById('register');
const loginBtn = document.getElementById('login');
const mobileRegister = document.getElementById('mobile-register');
const mobileLogin = document.getElementById('mobile-login');

const switchToSignUp = () => container.classList.add("active");
const switchToSignIn = () => container.classList.remove("active");

registerBtn.addEventListener('click', switchToSignUp);
loginBtn.addEventListener('click', switchToSignIn);

if (mobileRegister) mobileRegister.addEventListener('click', switchToSignUp);
if (mobileLogin) mobileLogin.addEventListener('click', switchToSignIn);

// function login() {
//     // Show a simple loading state on the button
//     const btn = event.target;
//     const originalText = btn.innerText;
//     btn.innerText = "Processing...";
//     btn.disabled = true;

//     setTimeout(() => {
//         window.location.href = "/home";
//     }, 800);
// }

// Social Login Integration
document.querySelectorAll('.social-icons a').forEach(button => {
    button.addEventListener('click', function (e) {
        e.preventDefault();
        const platform = this.classList.contains('social-google') ? 'Google' :
            this.classList.contains('social-facebook') ? 'Facebook' : 'Instagram';

        // Simulate a real social login popup
        const width = 500, height = 600;
        const left = (window.innerWidth / 2) - (width / 2);
        const top = (window.innerHeight / 2) - (height / 2);

        const popup = window.open('', 'SocialLogin', `width=${width},height=${height},left=${left},top=${top}`);

        if (popup) {
            popup.document.body.innerHTML = `
                <div style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                    <img src="/static/public/Icons/phone.svg" style="width: 50px; margin-bottom: 20px;">
                    <h2>Connecting to ${platform}...</h2>
                    <p>Please wait while we verify your credentials.</p>
                    <div style="margin-top: 30px; width: 30px; height: 30px; border: 4px solid #f3f3f3; border-top: 4px solid #2E7D32; border-radius: 50%; animation: spin 1s linear infinite; display: inline-block;"></div>
                    <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
                </div>
            `;

            setTimeout(() => {
                popup.close();
                window.location.href = "/home";
            }, 2000);
        }
    });
});
