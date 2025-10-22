# About

In this step usage of channel modelling with `vrtsim` will be explained

# Configure 

Modify the gNB config file at `targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf`. Add the following at the end:

```
channelmod = { 
  max_chan=10;
  modellist="modellist_rfsimu_1";
  modellist_rfsimu_1 = (
    {
        model_name                       = "server_tx_channel_model"
      	type                             = "AWGN";			  
      	ploss_dB                         = 0;
        noise_power_dB                   = 0; 
        forgetfact                       = 0;  
        offset                           = 0;      
        ds_tdl                           = 0;      
    }    
  );
};
```

# (Optional) Run UE with scope to visualize the impact

```
cmake ../../ -DENABLE_IMSCOPE=ON
```

```
cmake --build . --target imscope
```

Add `--imscope` to UE command line

# Run

```
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --device.name vrtsim --vrtsim.role server --vrtsim.chanmod 1
```

```
sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim
```

# Modifying the channel model

Try the following settings:

```
  modellist_rfsimu_1 = (
    {
        model_name                       = "server_tx_channel_model"
      	type                             = "Rayleigh8";			  
      	ploss_dB                         = 0;
        noise_power_dB                   = 0; 
        forgetfact                       = 0;  
        offset                           = 0;      
        ds_tdl                           = 0;      
    }    
  );
```
```
  modellist_rfsimu_1 = (
    {
        model_name                       = "server_tx_channel_model"
      	type                             = "AWGN";			  
      	ploss_dB                         = -5;
        noise_power_dB                   = 0; 
        forgetfact                       = 0;  
        offset                           = 0;      
        ds_tdl                           = 0;      
    }    
  );
```
```
  modellist_rfsimu_1 = (
    {
        model_name                       = "server_tx_channel_model"
      	type                             = "AWGN";			  
      	ploss_dB                         = -10;
        noise_power_dB                   = 0; 
        forgetfact                       = 0;  
        offset                           = 0;      
        ds_tdl                           = 0;      
    }    
  );
```

Add noise:

```
channelmod = { 
  max_chan=10;
  modellist="modellist_rfsimu_1";
  noise_power_dBFS = -42;
```
