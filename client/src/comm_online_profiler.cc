// update server load and communication status
#include "comm_online_profiler.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include "util.h"

/*Communication::Communication(struct tcp_info init)
  {
  current_tcp_info = init;
  }*/


double ca_next_cwnd(double cwnd)
{
        double ret = cwnd + (1/cwnd);
        return ret;
}

double bit_to_byte(uint64_t bit)
{
        return (double)bit / 8;
}

bool is_slowstart_phase(int cwnd, int ssthresh)
{
	if(cwnd >= ssthresh)
	{
		return false;
	}
	else
	{
		return true;
	}
}
double ss_next_cwnd(double cwnd, double ssthresh){
	double ret;
	ret = cwnd*2;
	if(ret >= ssthresh)
	{
		ret = ssthresh;
	}
	return ret;
}

//bottleneck_bw is Mbps
Communication::Communication(int bottleneck_bw)
{
	MAX_LINK = bottleneck_bw;
	LINK = bottleneck_bw;
	MAX_SERVER_CAPACITY = 1000;
	SERVER_CAPACITY = MAX_SERVER_CAPACITY;

	current_tcp_info.tcpi_snd_ssthresh = 99999999;
}
void Communication::Init(double rtt_ms)
{
	LINK = MAX_LINK;
	current_tcp_info.tcpi_rtt = rtt_ms * 1000;

}
void Communication::set_tcp_info(struct tcp_info tcpinfo)
{
	current_tcp_info = tcpinfo;
}
void Communication::print_info(bool verbose)
{
	if(verbose){
	std::cout <<"    tcpi_snd_cwnd: " << current_tcp_info.tcpi_snd_cwnd << std::endl;
	std::cout <<"         tcpi_rtt: " << current_tcp_info.tcpi_rtt << std::endl;
	std::cout <<"      tcpi_rttvar: " << current_tcp_info.tcpi_rttvar << std::endl;
	std::cout <<"tcpi_snd_ssthresh: " << current_tcp_info.tcpi_snd_ssthresh << std::endl;

	}
}

void Communication::get_server_capacity(std::vector<std::vector<int64_t>> shapes, std::vector<int> arrival_rate)
{
	double ret=0;
	for(int i = 0; i < shapes.size() ; i++)
	{
		int64_t s = vector_mul(shapes[i]) * 4 * 8 * 10;//bit
		ret += s * arrival_rate[i]; 
		std::cout << " s " << s << std::endl;
	}
	SERVER_CAPACITY = MAX_SERVER_CAPACITY - ret/1024/1024;

	std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SERVER CAPACITY " << SERVER_CAPACITY << std::endl;
}
int Communication::get_LINK(int64_t datasize_as_byte, double comm_time)
{
	
/*
	//std::cout << "comm " << comm_time <<std::endl;
	std::vector<double> y_list;
	std::vector<bool> reach_to_max_rtt_list;
	int reach_to_max_rtt_point = -1;
	for(int i = 10; i < MAX_LINK;i++)
	{
		bool reach_to_max_rtt;
		double y = expect_time_with_given_link(datasize_as_byte, i, &reach_to_max_rtt);
		y_list.push_back(y);
		reach_to_max_rtt_list.push_back(reach_to_max_rtt);
		if(reach_to_max_rtt && reach_to_max_rtt_point != -1)
		{
			reach_to_max_rtt_point = i;
		}
		//std::cout << i << " " <<  y <<  " " << reach_to_max_rtt << std::endl;
	}
	std::vector<double> err_list;

	for(int i = 0; i < y_list.size(); i++)
	{
		double err = std::abs(y_list[i]-comm_time); // ms
		err_list.push_back(err);
	}
	
	int min_index = std::min_element(err_list.begin(), err_list.end()) - err_list.begin();
	if(reach_to_max_rtt_point != -1)
	{
		return LINK;
	}
	else
	{
		return min_index;
	}	
	return min_index+10;
	
*/
	int ret = -1; 
	for(int l = 10; l < 100; l++){
		//std::cout << "cwnd " << current_tcp_info.tcpi_snd_cwnd << std::endl;
		double myrtt = ((double)(current_tcp_info.tcpi_snd_cwnd*1448*8)/(double)(l*1024*1024))*1000.0; //XXX 1000*1024, 100*1024
		double reported_rtt= current_tcp_info.tcpi_rtt / 1000.0; 
		//std::cout << "my rtt , reported rtt " << myrtt << " " << reported_rtt << std::endl;
		if(myrtt > reported_rtt)
		{
			ret = l;
		//	std::cout << l << " WALL " << std::endl;
		}
	}
	if(ret == -1)
	{
		return 99;
	}
	else
	{
		return ret;
	}
}
double Communication::expect_time_with_given_link(int64_t datasize_as_byte, int link, bool *reach_to_max_rtt, double measured_rtt)
{
	if (datasize_as_byte == 0)
	{
		return 0;
	}
	*reach_to_max_rtt = false;
	double link_capacity = 0;
	link_capacity = link*1024*1024; //XXX wired 1000
	double bdp_bit = link_capacity * (current_tcp_info.tcpi_rtt/1000);
	double bdp_byte = bit_to_byte(bdp_bit);
	double bdp_max_cwnd = bdp_byte/MSS;
	double count = 0;
	double expect_comm_time = 0;
	double num_seg = datasize_as_byte / MSS+1;
	double current_step_cwnd = 0;
	double sent = 0;
	double cumm_seg = 0;
	double candidate_cwnd = 0;
	double ret_expected_latency = 0;
	std::cout << "cuurent measued" << current_measured_rtt <<std::endl; 	


//	std::cout << "ssh " << current_tcp_info.tcpi_snd_ssthresh << std::endl;
	if(current_tcp_info.tcpi_snd_ssthresh == 0)
	{
		current_tcp_info.tcpi_snd_ssthresh = 30;//99999999;
	}
	while(num_seg>0)
	{
		//std::cout << "remein seg " << num_seg << std::endl;
		if(count == 0) // Send the first data
		{
			sent = init_cwnd;
			num_seg = num_seg - sent;
			current_step_cwnd = init_cwnd;
		}
		else
		{
			if(is_slowstart_phase(current_step_cwnd, current_tcp_info.tcpi_snd_ssthresh))
			{
				double candidate_cwnd = ss_next_cwnd(current_step_cwnd, current_tcp_info.tcpi_snd_ssthresh);
				
				
				if(bdp_max_cwnd > candidate_cwnd)
				{
					if (candidate_cwnd > num_seg)
					{
						candidate_cwnd = num_seg;
					}
					else
					{
						current_step_cwnd = candidate_cwnd;
				
					}
				}
				else if (candidate_cwnd > current_tcp_info.tcpi_snd_ssthresh)
				{
					if (candidate_cwnd > num_seg)
					{
						candidate_cwnd = num_seg;
					}
					else
						candidate_cwnd =  current_tcp_info.tcpi_snd_ssthresh;
				}
				else
				{
					 current_tcp_info.tcpi_snd_ssthresh = bdp_max_cwnd;
					if (candidate_cwnd > num_seg)
					{
						candidate_cwnd = num_seg;
					}
					else
						current_step_cwnd =  current_tcp_info.tcpi_snd_ssthresh;
				}
			
				current_step_cwnd = candidate_cwnd;
				sent = current_step_cwnd;
				num_seg = num_seg - sent;
				cumm_seg += sent;

				if(count > 1000)
				{
					exit(-1);
				}
			}
		
			else //congestion avoidance
			{
			//	is_ss = 0;
				double sent_packet = 0;
				
				double temp_cwnd = current_step_cwnd;
				while(1)
				{
					candidate_cwnd = ca_next_cwnd(temp_cwnd);
					sent_packet +=1;
					if( int(candidate_cwnd) > current_step_cwnd)
					{
						break;
					}
					temp_cwnd = candidate_cwnd;
				}
				num_seg = num_seg - sent_packet;
				current_step_cwnd = candidate_cwnd;
				cumm_seg += sent_packet;
			}
		}
		count +=1;

		current_step_cwnd = ceil(current_step_cwnd);
		//std::cout << "step cwnd " << count << "  " << current_step_cwnd <<" " << num_seg  << std::endl;
		double myrtt = ((current_step_cwnd*1448*8)/(link*1024*1024))*1000; //XXX 1000*1024, 100*1024
		double reported_rtt= measured_rtt;// current_tcp_info.tcpi_rtt / 1000.0; 
		double xrtt = 0;
		if(reported_rtt < myrtt)
		{
			xrtt = myrtt;
			*reach_to_max_rtt = true;
		}
		else
		{
			xrtt =  reported_rtt;
		}
		ret_expected_latency += xrtt;
		//ret_expected_latency += reported_rtt;
	
			std::cout << count << " " << current_step_cwnd << "," << current_tcp_info.tcpi_snd_ssthresh<< ","<<xrtt<<	std::endl;
	}
//	ret_expected_latency += current_tcp_info.tcpi_rtt / 1000.0; //3way handshake
//	ret_expected_latency += current_tcp_info.tcpi_rtt / 1000.0; //get result

	ret_expected_latency += measured_rtt;
ret_expected_latency += measured_rtt;
	
	expect_cwnd = current_step_cwnd;
	//std::cout << "\n" << datasize_as_byte << "  expected cwnd " << expect_cwnd << "count " << count << "ret " << ret_expected_latency << "rtt " << current_tcp_info.tcpi_rtt/1000.0 <<  std::endl;	
	return ret_expected_latency;
}

std::vector<double> Communication::expect_time_shapes(std::vector<std::vector<int64_t>> shapes, double link, ServerInfo server_info)
{
	std::vector<double> ret;
	double measured_rtt = server_info.GetServerInfoNoRefresh();
	std::cout << "measured rtt " << measured_rtt << std::endl;
	current_measured_rtt = measured_rtt;
	for (int i = 0; i < shapes.size() ; i++)
	{
		bool reach_to_max_rtt;
		double t = expect_time_with_given_link(vector_mul(shapes[i]) * 4 , link, &reach_to_max_rtt, measured_rtt);
		ret.push_back(t);
	}
	return ret;
}
