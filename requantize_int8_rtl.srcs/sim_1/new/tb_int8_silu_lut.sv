`timescale 1ns/1ps
module tb_int8_silu_lut;
    localparam int DATA_WIDTH=8, IDX_WIDTH=10, EXPECTED_CASES=3, EXPECTED_SEED=20260906;
    localparam string VEC="F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/silu_lut_vectors.txt";
    logic clk,rst_n,in_valid,in_ready,out_valid,out_ready;
    logic signed [DATA_WIDTH-1:0] in_data,out_data;
    logic [IDX_WIDTH-1:0] in_idx,out_idx;
    integer vf,tc,mc,hseed; string line;
    int8_silu_lut dut(.*);
    task send_in(input int signed v, input int idx);
        begin @(negedge clk); in_data=v; in_idx=idx; in_valid=1;
            while(!in_ready) @(negedge clk); @(posedge clk); @(negedge clk); in_valid=0; end endtask
    task recv_out(input int ei, input int signed ev, input int cid);
        begin out_ready=0; while(!out_valid) @(negedge clk);
            if(out_idx!==ei||$signed(out_data)!==ev) begin mc++; $display("FAIL c=%0d i=%0d rtl=%0d exp=%0d",cid,ei,out_data,ev); end
            else $display("PASS c=%0d i=%0d out=%0d",cid,ei,out_data);
            out_ready=1; @(posedge clk); @(negedge clk); out_ready=0; end endtask
    initial begin clk=0; forever #5 clk=~clk; end
    initial begin integer sc,cid,n,i; integer signed iv[0:511], ev[0:511]; tc=0; mc=0; rst_n=0; in_valid=0; out_ready=0;
        vf=$fopen(VEC,"r"); repeat(4) @(posedge clk); rst_n=1;
        while(!$feof(vf)) begin
            if($fgets(line,vf)==0) continue;
            if(line.len()==0) continue;
            if(line[0]=="#") begin $sscanf(line,"# seed=%d num_cases=%d",hseed,sc); continue; end
            if(line.substr(0,3)=="CASE") begin
                $sscanf(line,"CASE %d %d",cid,n);
                if($fgets(line,vf)==0||line!="IN\n") begin mc++; continue; end
                for(i=0;i<n;i++) begin $fgets(line,vf); $sscanf(line,"%d",iv[i]); end
                if($fgets(line,vf)==0||line!="EXPECT\n") begin mc++; continue; end
                for(i=0;i<n;i++) begin $fgets(line,vf); $sscanf(line,"%d",ev[i]); end
                for(i=0;i<n;i++) begin send_in(iv[i],i); recv_out(i,ev[i],cid); end
                tc++;
            end
        end
        $fclose(vf);
        if(hseed==EXPECTED_SEED&&tc==EXPECTED_CASES&&mc==0) $display("TEST RESULT: PASS %0d/%0d",tc,EXPECTED_CASES);
        else $display("TEST RESULT: FAIL tc=%0d mc=%0d",tc,mc);
        $finish;
    end
    initial begin #5000000; $display("TIMEOUT"); $finish; end
endmodule
