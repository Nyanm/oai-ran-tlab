#/*
# * Licensed to the EPYSYS SCIENCE (EpiSci) under one or more
# * contributor license agreements.
# * The EPYSYS SCIENCE (EpiSci) licenses this file to You under
# * the Episys Science (EpiSci) Public License (Version 1.1) (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      https://github.com/EpiSci/oai-lte-5g-multi-ue-proxy/blob/master/LICENSE
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about EPYSYS SCIENCE (EpiSci):
# *      bo.ryu@episci.com
# *-------------------------------------------------------------------------------
# * Modified by Russell Ford, Samsung Research America, russell.ford@samsung.com
# */

#include <sys/stat.h>
#include <sstream>
#include <sys/resource.h>
#include <csignal>
#include "proxy.h"
#include "lte_proxy.h"
#include "nr_proxy.h"

extern "C"
{
    int num_ues = 1;
}

static void show_usage();
// static bool is_numeric(const std::string&);
static void remove_log_file();
static bool is_ipaddress(const std::string &);

constexpr int DEFAULT_MAX_SECONDS = 10 * 60; // maximum run time

static const char *program_name;

static Multi_UE_NR_Proxy *nr_proxy_ptr;

static void die(const std::string& msg)
{
    std::clog << program_name << ": " << msg << std::endl;
    exit(EXIT_FAILURE);
}

static void try_help(const std::string& msg)
{
    die(msg + " (try --help)");
}

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  if (s != NULL) {
    printf("%s:%d %s() Exiting proxy: %s \n", file, line, function, s);
  }

  nr_proxy_ptr->stop_threads = true;

  if (assert) {
    abort();
  } else {
    sleep(1); // allow threads to exit first
    exit(EXIT_SUCCESS);
  }
}

void signal_handler(int sig) 
{
    printf("Got signal %s\n",strsignal(sig));
    exit_function(__FILE__, __FUNCTION__, __LINE__, "Proxy starting exit procedure\n", 0);  
}

int main(int argc, char *argv[])
{
    signal(SIGINT, signal_handler);  

    program_name = basename(argv[0]);

    // int max_seconds = DEFAULT_MAX_SECONDS;
    softmodem_mode_t softmodem_mode = SOFTMODEM_NR;
    std::vector<std::string> gnb_ipaddrs;
    std::vector<std::string> ue_ipaddrs;
    std::string proxy_ipaddr;
    std::string ch_trace_path;

    std::string arg{*++argv};
    while (--argc > 0)
    {
        if (arg == "--help") {
            show_usage();
            return EXIT_SUCCESS;
        }
        else if (arg == "--gnb") {
            while (argc-- > 0) {
                arg = *++argv;
                if (is_ipaddress(arg)) {
                    gnb_ipaddrs.push_back(arg);
                } else {
                    break;
                }
            }
            if (gnb_ipaddrs.size() == 0) {
                die("Need to provide at least one gNB IP address after --gnb");
            }
        }
        else if (arg == "--ue") {
            while (argc-- > 0) {
                arg = *++argv;
                if (is_ipaddress(arg)) {
                    ue_ipaddrs.push_back(arg);
                } else {
                    break;
                }
            }
            if (ue_ipaddrs.size() == 0) {
                die("Need to provide at least one UE IP address after --ue");
            }
        }
        else if (arg == "--proxy" && argc > 0) {
            arg = *++argv;
            argc--;
            if (is_ipaddress(arg)) {
                proxy_ipaddr = arg;
            }
            else {
                die("Need to provide proxy IP address after --proxy"); 
            }
            if (argc > 0)
                arg = *++argv;
        }
        else if (arg == "--lte") {
            // softmodem_mode = SOFTMODEM_LTE;
            // continue;
            die("LTE mode not currently supported.");
        }
        else if (arg == "--nr") {
            softmodem_mode = SOFTMODEM_NR;
            if (argc > 0)
                arg = *++argv;
        }
        else if (arg == "--nsa") {
            // softmodem_mode = SOFTMODEM_NSA;
            // continue;
            die("NSA mode not currently supported.");
        }
        else if (arg == "--ch" && argc > 0) {
            arg = *++argv;
            argc--;
            ch_trace_path = arg;
            if (argc > 0)
                arg = *++argv;
        }
        else {
            try_help("unexpected argument: " + arg);
        }
    }

    remove_log_file();

    /* This alarm is important because we run with the real-time scheduler.
       If (due to bugs) this process were to become run-away (running
       continuously without ever blocking), the alarm will eventually kill the
       process.  Otherwise, the host machine would need to be rebooted */
    // std::clog << "max_seconds: " << max_seconds << std::endl;
    // alarm(max_seconds);

    /* Enable core dumps */
    {
        struct rlimit lim = { RLIM_INFINITY, RLIM_INFINITY };
        if (setrlimit(RLIMIT_CORE, &lim) == -1)
            std::clog << program_name << ": setrlimit: " << strerror(errno) << '\n';
    }

    switch (softmodem_mode)
    {
    case SOFTMODEM_NR:
        {
            Multi_UE_NR_Proxy nr_proxy(gnb_ipaddrs, proxy_ipaddr, ue_ipaddrs, ch_trace_path.c_str());
            nr_proxy_ptr = &nr_proxy;
            nr_proxy.start(softmodem_mode);
        }
        break;
    // case SOFTMODEM_LTE:
    //     {
    //         Multi_UE_Proxy lte_proxy(ues, enb_ipaddr, proxy_ipaddr, ue_ipaddr);
    //         lte_proxy.start(softmodem_mode);
    //     }
    //     break;
    // case SOFTMODEM_NSA:
    //     {
    //         Multi_UE_Proxy lte_proxy(ues, enb_ipaddr, proxy_ipaddr, ue_ipaddr);
    //         Multi_UE_NR_Proxy nr_proxy(ues, gnb_ipaddr, proxy_ipaddr, ue_ipaddr);

    //         std::thread lte_th( &Multi_UE_Proxy::start, &lte_proxy, softmodem_mode);
    //         std::thread nr_th( &Multi_UE_NR_Proxy::start, &nr_proxy, softmodem_mode);

    //         lte_th.join();
    //         nr_th.join();
    //     }
    //     break;
    default:
        abort();
    }
    return EXIT_SUCCESS;
}

void show_usage()
{
    std::cout << "Usage: " << program_name << " --gnb gnb_ip1 ... gnb_ipN --proxy proxy_ip --ue ue_ip1 ... ue_ipN\n";
}

// bool is_numeric(const std::string &s)
// {
//     for (char c: s)
//     {
//         if (!isdigit((unsigned char) c))
//         {
//             return false;
//         }
//     }
//     return true;
// }

void remove_log_file()
{
    static const char log_name[] = "nfapi.log";
    if (remove(log_name) != 0 && errno != ENOENT)
    {
        std::clog << program_name << ": remove " << log_name
                  << ": " << strerror(errno) << std::endl;
    }
}

bool is_ipaddress(const std::string &s)
{
    sockaddr_in sa;
    return 1 == inet_pton(AF_INET, s.c_str(), &sa.sin_addr);
}
