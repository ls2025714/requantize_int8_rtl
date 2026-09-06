// ============================================================================
// 文件: tb_int8_residual_add.sv
// 阶段: D11
// 作用: 对拍饱和 INT8 残差加法（流式 x+y→z）
// DUT: int8_residual_add
// 向量: residual_add_vectors.txt（seed=20260905，5 case）
// CLI: scripts/run_residual_add_xsim.bat
// ============================================================================
//
// 流程: send_cmd(len) → stream X/Y → recv_z 对拍 z_idx/z_data
//
`timescale 1ns/1ps
module tb_int8_residual_add;
    localparam int DATA_WIDTH=8, IDX_WIDTH=10, EXPECTED_CASES=5, EXPECTED_SEED=20260905;
    localparam string VEC="F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/residual_add_vectors.txt";
    logic clk,rst_n,cmd_valid,cmd_ready; logic [IDX_WIDTH-1:0] cmd_len;
    logic x_valid,x_ready,y_valid,y_ready,z_valid,z_ready;
    logic signed [DATA_WIDTH-1:0] x_data,y_data,z_data;
    logic [IDX_WIDTH-1:0] z_idx;
    integer vf,tc,mc,hseed,hcases; string line;
    int8_residual_add dut(.*);
    task send_cmd(input int l); begin @(negedge clk); cmd_len=l; cmd_valid=1;
        while(!cmd_ready) @(negedge clk); @(posedge clk); @(negedge clk); cmd_valid=0; end endtask
    task send_x(input int signed v); begin @(negedge clk); x_data=v; x_valid=1;
        while(!x_ready) @(negedge clk); @(posedge clk); @(negedge clk); x_valid=0; end endtask
    task send_y(input int signed v); begin @(negedge clk); y_data=v; y_valid=1;
        while(!y_ready) @(negedge clk); @(posedge clk); @(negedge clk); y_valid=0; end endtask
    task recv_z(input int ei, input int signed ev, input int cid);
        begin z_ready=0; while(!z_valid) @(negedge clk);
            if(z_idx!==ei||$signed(z_data)!==ev) begin mc++; $display("FAIL c=%0d i=%0d rtl=%0d exp=%0d",cid,ei,z_data,ev); end
            else $display("PASS c=%0d i=%0d z=%0d",cid,ei,z_data);
            z_ready=1; @(posedge clk); @(negedge clk); z_ready=0; end endtask
    task run_case(input int cid, input int n);
        integer i, sc; integer signed xv[0:255], yv[0:255], ev[0:255]; string mk;
        begin
            for(i=0;i<n;i++) begin if($fgets(line,vf)==0) return;
                sc=$sscanf(line,"%d",xv[i]); if(sc!=1) mc++; end
            if($fgets(line,vf)==0||line!="Y\n") begin mc++; return; end
            for(i=0;i<n;i++) begin if($fgets(line,vf)==0) return;
                sc=$sscanf(line,"%d",yv[i]); if(sc!=1) mc++; end
            if($fgets(line,vf)==0||line!="EXPECT\n") begin mc++; return; end
            for(i=0;i<n;i++) begin if($fgets(line,vf)==0) return;
                sc=$sscanf(line,"%d",ev[i]); if(sc!=1) mc++; end
            send_cmd(n);
            for(i=0;i<n;i++) begin send_x(xv[i]); send_y(yv[i]); recv_z(i,ev[i],cid); end
            tc++; end endtask
    initial begin clk=0; forever #5 clk=~clk; end
    initial begin integer sc,cid,n; tc=0; mc=0; rst_n=0; cmd_valid=0; x_valid=0; y_valid=0; z_ready=0;
        vf=$fopen(VEC,"r"); repeat(4) @(posedge clk); rst_n=1;
        while(!$feof(vf)) begin
            if($fgets(line,vf)==0) continue;
            if(line.len()==0) continue;
            if(line[0]=="#") begin $sscanf(line,"# seed=%d num_cases=%d",hseed,hcases); continue; end
            if(line.substr(0,3)=="CASE") begin
                $sscanf(line,"CASE %d %d",cid,n);
                if($fgets(line,vf)==0||line!="X\n") mc++;
                else run_case(cid,n);
            end
        end
        $fclose(vf);
        if(hseed==EXPECTED_SEED&&tc==EXPECTED_CASES&&mc==0) begin
            $display("TEST RESULT: PASS %0d/%0d",tc,EXPECTED_CASES); end
        else $display("TEST RESULT: FAIL tc=%0d mc=%0d",tc,mc);
        $finish;
    end
    initial begin #5000000; $display("TIMEOUT"); $finish; end
endmodule
