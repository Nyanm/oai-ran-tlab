Differences between SCH_TTI.request and PFM

Header is the same, effectively FAPI header
then PFM has an offset of data followed by sfn, slot
SCH_TTI.request  after the header, has the sfn/slot, order which is shared to the other slot messages
SCH_TTI request then is followed by info regarding the algorithm to run, to which cell the message refers to and some more cell info (num Ues, num Antennas, etc)
PFM sort doesn't have any cell information, only related to UEs


struct cumac_pfm_tti_req_t {  
cumac_msg_header_t header;  
{
uint8_t message_count;  //!< Number of messages in this transmission  
uint8_t handle_id;      //!< handle_id is used as cell_id  
uint16_t type_id;       //!< Message type identifier (cumac_msg_t)  
uint32_t body_len;      //!< Length of message body in bytes  
uint8_t body[0];        //!< Variable-length message body (flexible array member)
}
uint32_t offset_ue_info_arr; // Offset of the PFM_UE_INFO data array in nvipc_buf->data_buf  
uint16_t sfn;  
uint16_t slot;  
uint16_t num_ue; // total number of UEs in the payload  
uint16_t num_output_sorted_lc[10];   
// number of output sorted LCs/LCGs per DL/UL QoS type, 0-9 are the indices of the QoS types:   
// 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical,   
// 5 - ul_gbr_critical, 6 - ul_gbr_non_critical, 7 - ul_ngbr_critical, 8 - ul_ngbr_non_critical, 9 - ul_mbr_non_critical.  
};

#define MAX_NUM_LCG 4  
#define MAX_NUM_LC 4
struct PFM_UE_INFO {  
PFM_DL_LC_INFO      dl_lc_info[MAX_NUM_LC];  
PFM_UL_LCG_INFO     ul_lcg_info[MAX_NUM_LCG];  
uint32_t            ambr;  
uint32_t            rcurrent_dl;  
uint32_t            rcurrent_ul;  
uint16_t            rnti;  
uint8_t             num_layers_dl;  
uint8_t             num_layers_ul;  
uint8_t             flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - is_scheduled_dl, flags & 0x04 - is_scheduled_ul  
uint8_t             carrier_id;  
uint8_t             num_dl_lcs;  
uint8_t             num_ul_lcgs;  
};

struct PFM_DL_LC_INFO {  
uint32_t        tbs_scheduled;  
uint32_t        ravg;  
uint32_t        pfm;   
uint8_t         flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - reset ravg to 1.0  
uint8_t         qos_type; // 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical  
uint8_t         padding[2];  // Padding to align to 32-bit  
};

struct PFM_UL_LCG_INFO {  
uint32_t        tbs_scheduled;  
uint32_t        ravg;  
uint32_t        pfm;   
uint8_t         flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - reset ravg to 1.0  
uint8_t         qos_type; // 0 - ul_gbr_critical, 1 - ul_gbr_non_critical, 2 - ul_ngbr_critical, 3 - ul_ngbr_non_critical, 4 - ul_mbr_non_critical   
uint8_t         padding[2];  // Padding to align to 32-bit  
};
