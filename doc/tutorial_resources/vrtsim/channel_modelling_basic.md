# About

In this step usage of channel modelling with `vrtsim` will be explained

# Running 

1. Modify the gNB config file at `targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf`. Add the following at the end:

```
channelmod = { 
  max_chan=10;
  modellist="modellist_rfsimu_1";
  modellist_rfsimu_1 = (
    {
        model_name                       = "client_tx_channel_model"
      	type                             = "AWGN";			  
      	ploss_dB                         = 5;
        noise_power_dB                   = -10; 
        forgetfact                       = 0;  
        offset                           = 0;      
        ds_tdl                           = 0;      
    },
    {
        model_name                       = "server_tx_channel_model"
      	type                             = "AWGN";			  
      	ploss_dB                         = 5;
        noise_power_dB                   = -10; 
        forgetfact                       = 0;  
        offset                           = 0;      
        ds_tdl                           = 0;      
    }    
  );
};
```