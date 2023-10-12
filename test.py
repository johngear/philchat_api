import requests
import time
import argparse
from back.config import EC2_API_URL, EC2_API_IP, PYTHON_ANYWHERE_URL


def main(mode, api_url, model, temp, num_requests):

    # URLs of the API endpoints
    if api_url == 'local':
        url = 'http://127.0.0.1:8000/q_and_a'
    elif api_url == 'url_ec2':
        url = EC2_API_URL + 'q_and_a'
    elif api_url == 'ip_ec2':
        url = EC2_API_IP + 'q_and_a'
    elif api_url == 'python_anywhere':
        url = PYTHON_ANYWHERE_URL + 'q_and_a'
    else:
        raise ValueError("Bad URL Mode")

    print(f"Mode                                : {mode}")
    print(f"URL                                 : {api_url}")
    print(f"API Endpoint                        : {url}")
    print(f"Model                               : {model}")
    print(f"Temp                                : {temp}")

    if mode == 'timing': 
        print(f"Number of Requests                  : {num_requests}")
    print("\n")

    if mode == "short_q":
        q = "Quick, name a philosopher"
        make_request(get_request(q, temp, model),url)
        
    elif mode == "long_q":
        q = "Who was Kant? Give a long answer"
        make_request(get_request(q, temp, model),url)

    elif mode == "timing":
        start = time.time()
        data = make_big_request(model, temp, num_requests)
        make_request(data,url, False)
        print(f'Time for processing {num_requests} requests      : {(time.time() - start):.2f} seconds')

    elif mode == "health":
        response = requests.get(url.rstrip("q_and_a") +"health")
        print(response)     
    else:
        raise ValueError("Bad Mode Input")
    print("\n")

def get_request(q, temp, model):
    return [{
        'q': q,
        'temp': temp,
        'model': model
    }]

def make_request(data, url, do_print=True):

    for d in data:
        # Send the POST request
        response = requests.post(url, json=d)
        # Check the response status code
        if response.status_code == 200:
            # Request was successful
            result = response.json()['result']
            if do_print: print(result)
        else:
            # Request failed
            print('Request failed with status code:', response.status_code)

def make_big_request(model, temp, num_requests=3):
    full_data = []

    example_queries = [
        "What is the problem of induction?",
        "Who is Hume?",
        "What is the meaning of life?",
        "Explain Godels incompleteness theorems",
        "Who was Carnap?",
        "Explain John Rawls' Theory of Justice",
        "Hi. Say it back!",
        "What is the Original Position",
        "Who was Aristotle",
        "Who were the logical positivists?",
    ]
    for q in example_queries[:num_requests]:
        full_data.append(get_request(q,temp,model)[0])

    return full_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various API performance.')

    parser.add_argument('-m', '--mode', default='short_q', choices=['short_q', 'long_q', 'timing', 'health'], help='Which questions to ask the API.')
    parser.add_argument('-u', '--url', default='url_ec2', choices=['local', 'ip_ec2', 'url_ec2', 'python_anywhere'], help='Which API URL to access.')
    parser.add_argument('-c', '--model', default='chat', choices=['chat', 'completion'], help='Call the OpenAI API in Chat or Completion mode')
    parser.add_argument('-n', '--num_requests', default=3, type=int, choices=range(1, 11),help='If doing Timing, how many requests to make')  
    parser.add_argument('-t', '--temp', default=0.2, type=float, help='Temperature of the model.')  
    
    args = parser.parse_args()

    main(args.mode, args.url, args.model, args.temp, args.num_requests)
