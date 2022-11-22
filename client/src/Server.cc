#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include "util.h"
extern "C" {
#include<curl/curl.h>
}


enum {
    ERROR_ARGS = 1,
    ERROR_CURL_INIT = 2
};

enum {
    OPTION_FALSE = 0,
    OPTION_TRUE = 1
};

enum {
    FLAG_DEFAULT = 0
};

#include "Server.h"


#define SERVER_STATUS_URL "http://210.107.197.107:8004"
#define SERVER_STATUS_PATH "/status"

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string *) userp)->append((char *) contents, size * nmemb);
    return size * nmemb;
}

ServerInfo::ServerInfo()
{
	rtt = 0;
	link_capacity = 100;
	sf = 1;
	queueing = 0;
	server_information_refresh_time = 0;
};
void ServerInfo::init()
{
	sf= 1;
	queueing = 0;
	last_batch = 0;
	last_server_infertime = 0;
}

void ServerInfo::RTTrefresh(double current_rtt)
{
	
	if(rtt_history.size() !=10)
	{
		rtt_history.push_back(current_rtt);
	}		
	else
	{
		rtt_history.erase(rtt_history.begin());
		rtt_history.push_back(current_rtt);
	}
	rtt = std::accumulate(rtt_history.begin(), rtt_history.end(),0.0)/(double)rtt_history.size();

}
void ServerInfo::GetServerInfo()
{
	std::string readBuffer;

	CURL *ctx = curl_easy_init();
	std::string status_url = std::string(SERVER_STATUS_URL)+std::string(SERVER_STATUS_PATH);
	curl_easy_setopt(ctx, CURLOPT_URL, status_url.c_str());
	curl_easy_setopt(ctx, CURLOPT_NOPROGRESS, OPTION_TRUE);
	curl_easy_setopt(ctx, CURLOPT_WRITEFUNCTION, WriteCallback);
	curl_easy_setopt(ctx, CURLOPT_WRITEDATA, (void *) &readBuffer);
	curl_easy_setopt(ctx, CURLOPT_TCP_NODELAY, 1L);

	std::chrono::steady_clock::time_point getstatus_start = std::chrono::steady_clock::now();
	const CURLcode rc = curl_easy_perform(ctx);
	std::chrono::steady_clock::time_point getstatus_end = std::chrono::steady_clock::now();

	double current_rtt = std::chrono::duration_cast<std::chrono::microseconds>(getstatus_end - getstatus_start).count()/1000.0 /2; 
	
	RTTrefresh(current_rtt);
	if (CURLE_OK != rc) {
		std::cerr << "Error from cURL: " << curl_easy_strerror(rc) << std::endl;
	}

	// cleanup
	curl_easy_cleanup(ctx);
	curl_global_cleanup();

	std::string delimiter = ",";
	std::vector<std::string> parsed;
	size_t pos= 0;
	std::string temp;
	while ((pos = readBuffer.find(delimiter)) != std::string::npos) {
		temp = readBuffer.substr(0, pos);
		parsed.push_back(temp);
		//std::string strtoken(queue_status);
		readBuffer.erase(0, pos + delimiter.length());
	}
	
	parsed.push_back(readBuffer);

	percentile.clear();
	server_information_send_time = std::stol(parsed[0]);
	server_information_update_time = std::stol(parsed[1]);
	server_information_refresh_time = get_current_unixtime();
	average_interval = std::stod(parsed[2]);
	average_throughput = std::stod(parsed[3]);
	average_batch = std::stod(parsed[4]);
	average_infer_time = std::stod(parsed[5]);
        average_queue_time = std::stod(parsed[6]);	
	for(int i = 7; i < 17; i++)
	{
		percentile.push_back(std::stod(parsed[i]));
	}
	CURRENT_SERVER_CAPACITY = std::stoi(parsed[17]);
	

	if(isServerInfoExpired())
	{
		isServerInfoExpiredResult = true;
		ResetServerInfo();
	}
	std::cout << "--------------Get information result ----------------" << std::endl;	
	std::cout << "average interval   : " << average_interval<< std::endl;
        std::cout << "average throughput : " << average_throughput << std::endl;
        std::cout << "average batch      : " << average_batch << std::endl;
	std::cout << "average infer time : " << average_infer_time <<  std::endl;
	std::cout << "average queue time : " << average_queue_time << std::endl;
	std::cout << "time diff          : " << server_information_send_time - server_information_update_time << std::endl;
	std::cout << "CURRENT LINK       : " << CURRENT_SERVER_CAPACITY << std::endl;
	std::cout << "Server Info expired: " << isServerInfoExpiredResult << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;


		server_information_refresh_time = get_current_unixtime();
}

double ServerInfo::GetServerInfoNoRefresh()
{
	std::string readBuffer;

	CURL *ctx = curl_easy_init();
	std::string status_url = std::string(SERVER_STATUS_URL)+std::string(SERVER_STATUS_PATH);
	curl_easy_setopt(ctx, CURLOPT_URL, status_url.c_str());
	curl_easy_setopt(ctx, CURLOPT_NOPROGRESS, OPTION_TRUE);
	curl_easy_setopt(ctx, CURLOPT_WRITEFUNCTION, WriteCallback);
	curl_easy_setopt(ctx, CURLOPT_WRITEDATA, (void *) &readBuffer);
	curl_easy_setopt(ctx, CURLOPT_TCP_NODELAY, 1L);

	std::chrono::steady_clock::time_point getstatus_start = std::chrono::steady_clock::now();
	const CURLcode rc = curl_easy_perform(ctx);
	std::chrono::steady_clock::time_point getstatus_end = std::chrono::steady_clock::now();
	double current_rtt = std::chrono::duration_cast<std::chrono::microseconds>(getstatus_end - getstatus_start).count()/1000.0 /2; 
	current_measured_rtt = current_rtt;
	return current_rtt;
}



void ServerInfo::ResetServerInfo()
{
	average_infer_time = 0;
	average_queue_time = 0;
	std::fill(percentile.begin(), percentile.end(), 0);
}
void ServerInfo::SetServerInfo(double a, double b, double c, double d, double e)
{
	average_interval = a;
	average_throughput = b;
	average_batch = c;
	average_infer_time = d;
	average_queue_time = e;
}

bool ServerInfo::isMyInfoExpired()
{
	uint64_t current_time = get_current_unixtime();
	if(server_information_refresh_time+1000 < current_time )
	{
		return true;
	}
	
	return false;
}
bool ServerInfo::isServerInfoExpired()
{
	if(server_information_send_time > server_information_update_time + 1000000)
	{
		return true;
	}
	
	return false;
}
