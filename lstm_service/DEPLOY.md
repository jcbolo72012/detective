# Deploy the LSTM forecast service on AWS Lambda

Cost: pay per call (a forecast is ~10-30 s at 2 GB, well under a cent). Nothing runs when idle.
Easiest path on Windows: **AWS CloudShell** (browser terminal with Docker and your credentials already set up).
Untested by the author: expect to adjust a flag or two. Check the AWS docs if a command is rejected.

## 1. Open CloudShell and get the code
    git clone https://github.com/jcbolo72012/detective.git
    cd detective/lstm_service

## 2. Variables
    export AWS_REGION=us-east-1
    export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    export REPO=trend-lstm
    export TOKEN=$(openssl rand -hex 24)      # shared secret; save it, the app needs it
    echo $TOKEN

## 3. Build and push the image (CPU-only PyTorch)
    aws ecr create-repository --repository-name $REPO --region $AWS_REGION
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
    docker build --platform linux/amd64 --provenance=false -t $REPO .
    docker tag $REPO:latest $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
    docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest

## 4. Role and function
    aws iam create-role --role-name trend-lstm-role --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    aws iam attach-role-policy --role-name trend-lstm-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    sleep 10
    aws lambda create-function --function-name trend-lstm --package-type Image --code ImageUri=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest --role arn:aws:iam::$ACCOUNT:role/trend-lstm-role --memory-size 2048 --timeout 60 --environment "Variables={FORECAST_TOKEN=$TOKEN}" --region $AWS_REGION

## 5. Limit abuse, then expose a URL
    aws lambda put-function-concurrency --function-name trend-lstm --reserved-concurrent-executions 2 --region $AWS_REGION
    aws lambda create-function-url-config --function-name trend-lstm --auth-type NONE --region $AWS_REGION
    aws lambda add-permission --function-name trend-lstm --statement-id url-public --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE --region $AWS_REGION
    aws lambda add-permission --function-name trend-lstm --statement-id url-invoke --action lambda:InvokeFunction --principal "*" --invoked-via-function-url --region $AWS_REGION

(The URL is public, but every request must carry the secret in the `x-forecast-token` header, inputs are
capped in forecast.py, and at most 2 run at once.) The create-function-url-config output contains `FunctionUrl`.

## 6. Test
    curl -s -X POST "<FunctionUrl>" -H "x-forecast-token: $TOKEN" -d '{}'          # expect {"error": ...} 400
    curl -s -X POST "<FunctionUrl>" -d '{}'                                         # expect 401

## 7. Point the app at it
Streamlit Cloud -> Settings -> Secrets:

    FORECAST_URL = "<FunctionUrl>"
    FORECAST_TOKEN = "<the token>"

## Guardrails
- Add an AWS Budgets alert (e.g. $5/month) in the console.
- First call after idle is slower (cold start, ~5-10 s) and the app waits up to 75 s.
- Update the code: rebuild/push the image, then `aws lambda update-function-code --function-name trend-lstm --image-uri <uri>`.
- Delete everything after the event: `aws lambda delete-function ...`, `aws ecr delete-repository --force ...`.
