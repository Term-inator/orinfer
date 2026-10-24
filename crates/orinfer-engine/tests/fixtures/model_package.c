/* Cross-language ABI fixture: deliberately a model family unknown to core. */
#include "orinfer_model.h"
#include <stdlib.h>
#include <string.h>
static int32_t bytes(OrinferOwnedBytes *out, const char *s) {
    out->len=strlen(s); out->data=malloc(out->len); memcpy(out->data,s,out->len); return 0;
}
static int32_t describe(OrinferOwnedBytes *out) {
    return bytes(out,"{\"package\":\"test-model\",\"version\":\"1\",\"target\":\"sm_87\",\"runtime_abi\":1,\"architectures\":[\"test_family\"],\"compute_policies\":[\"test_policy\"]}");
}
static int32_t create(const uint8_t *p,size_t n,void **model,OrinferOwnedBytes *out) {
    char *input=malloc(n+1);memcpy(input,p,n);input[n]=0;
    char *metadata=strstr(input,"\"metadata\":");
    if(!metadata) {free(input);bytes(out,"missing metadata");return 1;}
    metadata=strchr(metadata,'{'); char *end=metadata;
    int depth=0, quoted=0, escaped=0;
    for(;*end;++end) {
        char c=*end;
        if(quoted) {if(escaped)escaped=0;else if(c=='\\')escaped=1;else if(c=='"')quoted=0;}
        else {if(c=='"')quoted=1;else if(c=='{')++depth;else if(c=='}' && --depth==0){++end;break;}}
    }
    if(depth!=0) {free(input);bytes(out,"invalid metadata");return 1;}
    size_t count=(size_t)(end-metadata);
    const char *programs="\"programs\":{}";
    char *slot=strstr(metadata,programs);
    if(!slot || slot>=end) {free(input);bytes(out,"missing program slot");return 1;}
    const char *replacement="\"programs\":{\"decode\":[{\"kind\":\"kernel\",\"name\":\"decode\"}],\"prefill\":[{\"kind\":\"kernel\",\"name\":\"prefill\"}],\"head\":[{\"kind\":\"zero\",\"destination\":\"Scratch\",\"bytes\":4}]}";
    if (strstr(input,"\"catalog\":") != NULL) replacement = programs;
    const char *prefix="{\"metadata\":";
    const char *suffix=",\"decode_programs\":[\"decode\"]}";
    size_t left=(size_t)(slot-metadata),right=count-left-strlen(programs);
    out->len=strlen(prefix)+left+strlen(replacement)+right+strlen(suffix);
    out->data=malloc(out->len);uint8_t *cursor=out->data;
    memcpy(cursor,prefix,strlen(prefix));cursor+=strlen(prefix);
    memcpy(cursor,metadata,left);cursor+=left;
    memcpy(cursor,replacement,strlen(replacement));cursor+=strlen(replacement);
    memcpy(cursor,slot+strlen(programs),right);cursor+=right;
    memcpy(cursor,suffix,strlen(suffix));
    *model=malloc(1);free(input);return 0;
}
static void destroy(void *model) {free(model);}
static void free_bytes(OrinferOwnedBytes b) {free(b.data);}
static void free_batch(OrinferBatch b) {(void)b;}
static void free_visual(OrinferVisual v) {(void)v;}
#define SPAN(s) {(const uint8_t *)(s),sizeof(s)-1}
static const OrinferArgument args[]={{2,3}};
static const OrinferView views[]={{SPAN("Input"),SPAN("Scratch"),128}};
static const OrinferCommand commands[]={{0,SPAN("projection"),SPAN(""),0,7,1,{3,1,1},args,1,views,1}};
static OrinferBinding bindings[]={{0,0,SPAN("")},{1,7,SPAN("PrivateState")}};
static int32_t batch(void *model,const OrinferSegment *s,size_t n,uint32_t plan,OrinferBatch *out,OrinferOwnedBytes *error) {
    (void)model; memset(out,0,sizeof(*out));
    bindings[1].sequence=n?s[0].slot:7;
    out->commands=commands;out->command_count=(plan & ORINFER_BATCH_INCLUDE_PLAN)?1:0;out->bindings=bindings;out->binding_count=2;
    return bytes(error,"");
}
static int32_t visual(void *model,const uint32_t *t,size_t n,const OrinferImageGrid *g,size_t m,size_t c,OrinferVisual *out,OrinferOwnedBytes *error) {
    (void)model;(void)t;(void)n;(void)g;(void)m;(void)c;(void)out; bytes(error,"unsupported");return 1;
}
static const OrinferModelApi api={ORINFER_MODEL_ABI,sizeof(OrinferModelApi),describe,create,destroy,batch,free_bytes,free_batch,visual,free_visual};
const OrinferModelApi *orinfer_model_v1(uint32_t version) {return version==ORINFER_MODEL_ABI?&api:NULL;}

#ifdef ORINFER_PLAN_FIXTURE_HEADER
#include ORINFER_PLAN_FIXTURE_HEADER
#else
static const OrinferPlanArgument plan_args[] = {
    {0, SPAN("Input"), 0, 0}, {1, SPAN("State"), 64, 0},
    {2, SPAN(""), 0, UINT32_MAX}, {3, SPAN(""), 0, UINT32_MAX},
    {4, SPAN(""), 0, UINT64_MAX}, {5, SPAN(""), 0, UINT64_MAX},
    {6, SPAN(""), 0, 0x80000000}
};
static const OrinferPlanKernel plan_kernels[] = {{SPAN("projection"), 0, {1,1,1}, plan_args, 7, SIZE_MAX, 1}};
static const OrinferPlanCommand plan_commands[] = {
    {0, SPAN("projection"), SPAN(""), 0},
    {1, SPAN("Input"), SPAN("State"), 4},
    {2, SPAN(""), SPAN("State"), 4}
};
static const OrinferPlanProgram plan_programs[] = {{SPAN("decode"), plan_commands, 3}};
#endif
static int32_t export_plans(void *model, OrinferPlans *out, OrinferOwnedBytes *error) {
    (void)model;
    *out = (OrinferPlans){NULL, plan_kernels, sizeof(plan_kernels)/sizeof(plan_kernels[0]), plan_programs, sizeof(plan_programs)/sizeof(plan_programs[0])};
    return bytes(error, "");
}
static void free_plans(OrinferPlans p) {(void)p;}
static const OrinferPlanApi plan_api = {1, sizeof(OrinferPlanApi), sizeof(OrinferPlanKernel), sizeof(OrinferPlanArgument), sizeof(OrinferPlanProgram), sizeof(OrinferPlanCommand), export_plans, free_plans};
const OrinferPlanApi *orinfer_model_plans_v1(uint32_t version) {return version == 1 ? &plan_api : NULL;}

typedef struct { uint32_t length; OrinferInputUpload upload; } InputStorage;
static int32_t prepare_inputs(void *model, const uint32_t *tokens, size_t count,
    const uint32_t *history, size_t history_count, OrinferInputs *out, OrinferOwnedBytes *error) {
    (void)model; (void)tokens; (void)history; (void)history_count;
    InputStorage *storage = malloc(sizeof(*storage));
    storage->length = (uint32_t)count;
    storage->upload = (OrinferInputUpload){SPAN("SegmentLength"),
        {(const uint8_t *)&storage->length, sizeof(storage->length)}};
    *out = (OrinferInputs){storage, &storage->upload, 1};
    return bytes(error, "");
}
static int32_t prepare_program(void *model, OrinferBytes program, const uint32_t *tokens, size_t count,
    const uint32_t *history, size_t history_count, OrinferInputs *out, OrinferOwnedBytes *error) {
    (void)program;
    return prepare_inputs(model, tokens, count, history, history_count, out, error);
}
static void free_inputs(OrinferInputs input) {free(input.owner);}
static const OrinferInputApi input_api = {1, sizeof(OrinferInputApi), prepare_inputs, free_inputs, prepare_program};
const OrinferInputApi *orinfer_model_inputs_v1(uint32_t version) {return version == 1 ? &input_api : NULL;}
