# Heroku Deployment Guide

## Prerequisites

1. **Heroku Account**: Sign up at https://www.heroku.com/
2. **Heroku CLI**: Download from https://devcenter.heroku.com/articles/heroku-cli
3. **Git**: Install from https://git-scm.com/

## Step-by-Step Deployment

### 1. Initialize Git Repository

```bash
cd d:\Downloads_D\customer_churn_prediction
git init
git add .
git commit -m "Initial commit - Customer Churn Prediction API"
```

### 2. Create Heroku App

```bash
heroku login
heroku create your-app-name
```

Replace `your-app-name` with a unique name (e.g., `churn-prediction-api-2024`). If you skip this, Heroku will generate one.

Get your app name:
```bash
heroku apps:info
```

### 3. Set Environment Variables (if needed)

```bash
heroku config:set FLASK_DEBUG=False
```

### 4. Deploy to Heroku

```bash
git push heroku main
```

If your branch is `master` instead of `main`, use:
```bash
git push heroku master
```

### 5. Verify Deployment

```bash
heroku open
```

This opens your app in the browser. You should see the API home page.

Or test via curl:
```bash
curl https://your-app-name.herokuapp.com/
```

### 6. View Logs

```bash
heroku logs --tail
```

To see more recent logs:
```bash
heroku logs -n 100
```

---

## Testing Your Deployed API

Once deployed, test the `/predict` endpoint with your app URL:

### PowerShell Example:
```powershell
$body = @{
    tenure = 24
    MonthlyCharges = 79.85
    TotalCharges = 1916.4
    gender = 1
    SeniorCitizen = 0
    Partner = 1
    Dependents = 0
    PhoneService = 1
    MultipleLines = 0
    PaperlessBilling = 1
    OnlineSecurity_Yes = 1
    OnlineBackup_Yes = 0
    Contract_One_year = 0
    Contract_Two_year = 0
    CLV = 1916.4
    MonthlyCost_ratio = 0.042
    tenure_years = 2.0
    is_high_value = 0
    is_new_customer = 0
    is_long_term = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://your-app-name.herokuapp.com/predict" -Method Post -Body $body -ContentType "application/json"
```

### Bash/cURL Example:
```bash
curl -X POST https://your-app-name.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "MonthlyCharges": 79.85,
    "TotalCharges": 1916.4,
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "PaperlessBilling": 1,
    "OnlineSecurity_Yes": 1,
    "OnlineBackup_Yes": 0,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "CLV": 1916.4,
    "MonthlyCost_ratio": 0.042,
    "tenure_years": 2.0,
    "is_high_value": 0,
    "is_new_customer": 0,
    "is_long_term": 0
  }'
```

---

## Files Created for Deployment

- **Procfile**: Tells Heroku how to run the app (uses Gunicorn)
- **runtime.txt**: Specifies Python version (3.11.7)
- **.gitignore**: Excludes unnecessary files from git
- **requirements.txt**: Updated with `gunicorn` and `requests`
- **app.py**: Modified to use PORT from environment variable

---

## Common Issues & Solutions

### Issue: "cannot find module models"
**Solution**: Ensure `Procfile` uses `app:app` correctly. Models should be in `models/` directory.

### Issue: "ModuleNotFoundError"
**Solution**: Run `pip install -r requirements.txt` locally first. Push changes to git.

### Issue: Build takes too long
**Solution**: Heroku slug size limit is 500MB. Consider removing TensorFlow if not used:
```bash
# Edit requirements.txt and remove or comment out tensorflow==2.13.0
git add requirements.txt
git commit -m "Remove TensorFlow to reduce slug size"
git push heroku main
```

### Issue: "H14 - No web processes running"
**Solution**: Check logs with `heroku logs --tail`. Ensure `Procfile` is correct.

---

## Update Deployment

After making changes locally:

```bash
git add .
git commit -m "Update description of changes"
git push heroku main
```

---

## Monitor Your App

### View Real-time Logs
```bash
heroku logs --tail
```

### Check App Status
```bash
heroku ps
```

### Restart App
```bash
heroku restart
```

### Scale Dynos (if on paid plan)
```bash
heroku ps:scale web=2
```

---

## Useful Heroku Commands

```bash
# View all apps
heroku apps

# Switch between apps
heroku apps:info -a app-name

# Set config variables
heroku config:set KEY=value

# Remove config variables
heroku config:unset KEY

# Destroy app (careful!)
heroku apps:destroy --app your-app-name
```

---

## Next Steps

1. **Custom Domain**: Add a domain name to your app in Heroku dashboard
2. **Monitoring**: Set up monitoring/alerts in Heroku dashboard
3. **CI/CD**: Connect GitHub for automatic deployments on push
4. **Database**: If you need persistent storage, consider Heroku Postgres

---

## Support

- Heroku Docs: https://devcenter.heroku.com/
- Python on Heroku: https://devcenter.heroku.com/articles/getting-started-with-python

