import sys
from itertools import islice
from pathlib import Path
 
# 4//U V W X Y
# 3//P Q R S T
# 2//K L M N O
# 1//F G H I J
# 0//A B C D E
#  //0 1 2 3 4

#at 0+0.5 0+0.5 -> RSMN with M being the center and value .w

# //gather order
# // 1 x y
# // 0 w z
# //   0 1

#Optimized luma texture loader (4x faster)
gather_5x5='''float pixel[5][5];
const vec4 ABFG = HOOKED_gather(pos + vec2(-2+0.5,-2+0.5)*HOOKED_pt,0);
pixel[0][0] = ABFG.w; //A
pixel[1][0] = ABFG.z; //B
pixel[0][1] = ABFG.x; //F
pixel[1][1] = ABFG.y; //G
const vec4 CDHI = HOOKED_gather(pos + vec2(+0+0.5,-2+0.5)*HOOKED_pt,0);
pixel[2][0] = CDHI.w; //C
pixel[3][0] = CDHI.z; //D
pixel[2][1] = CDHI.x; //H
pixel[3][1] = CDHI.y; //I
const vec4 KLPQ = HOOKED_gather(pos + vec2(-2+0.5,0+0.5)*HOOKED_pt,0);
pixel[0][2] = KLPQ.w; //K
pixel[1][2] = KLPQ.z; //L
pixel[0][3] = KLPQ.x; //P
pixel[1][3] = KLPQ.y; //Q
const vec4 MNRS = HOOKED_gather(pos + vec2(+0+0.5,+0+0.5)*HOOKED_pt,0);
float true_center = float(HOOKED_texOff(vec2(0,0))); // 
pixel[2][2] = MNRS.w-true_center ; //M
pixel[2][2] = MNRS.w; //M
pixel[3][2] = MNRS.z; //N
pixel[2][3] = MNRS.x; //R
pixel[3][3] = MNRS.y; //S

const vec4 EZJZ = HOOKED_gather(pos + vec2(+2+0.5,-2+0.5)*HOOKED_pt,0);
pixel[4][0] = EZJZ.w; //E
pixel[4][1] = EZJZ.x; //J
const vec4 OZTZ = HOOKED_gather(pos + vec2(+2+0.5,+0+0.5)*HOOKED_pt,0);
pixel[4][2] = OZTZ.w; //O
pixel[4][3] = OZTZ.x; //T
const vec4 UVZZ = HOOKED_gather(pos + vec2(-2+0.5,+2+0.5)*HOOKED_pt,0);
pixel[0][4] = UVZZ.w; //U
pixel[1][4] = UVZZ.z; //V
const vec4 WXZZ = HOOKED_gather(pos + vec2(0+0.5,+0+0.5)*HOOKED_pt,0);
pixel[2][4] = UVZZ.w; //W
pixel[3][4] = UVZZ.z; //X
//Y
pixel[4][4] = HOOKED_texOff(vec2(2,2)).x;
'''




def read_weights(filename, phrase, size=1):

    ln = None
    with open(filename) as f:
        for i, line in enumerate(f, 1):
            if phrase in line:
                ln = i
                break
    if ln == None:
        print(f'Failed to find weights "{phrase}" in {filename}.')
        sys.exit(1)
        return False

    content = []
    with open(filename) as f:
        for line in islice(f, ln, ln + size):
            if line.find('[') != -1:
                line = line[line.index('[') + 1:]
            if line.find(']') != -1:
                line = line[:line.rindex(']')]
            content.append(line)

    return [x.strip() for x in content]

def format_weights(weights, n, length=4):
    return ",".join(['{:.16f}'.format(float(i)) for i in weights.strip(",").split(",")[n:n+length]])

def base_header(file,scale):
    file.write('//!HOOK LUMA\n')
    file.write('//!WHEN OUTPUT.w LUMA.w / {0}.100 > OUTPUT.h LUMA.h / {0}.100 > *\n'.format(scale - 1))

def header_feature(file, scale, thread_count, vec4_per_pixel):
    base_header(file,scale)
    file.write(f'//!DESC feature map\n')
    file.write(f'//!COMPUTE {vec4_per_pixel*thread_count} {thread_count} {thread_count} {thread_count}\n')
    file.write(f'//!BIND LUMA\n')
    file.write(f'//!SAVE FEATURE\n')
    file.write(f'//!COMPONENTS 4\n')
    file.write(f'//!WIDTH LUMA.w {vec4_per_pixel} *\n')
    file.write(f'//!HEIGHT LUMA.h\n')

def header_mapping(file, scale, thread_count, vec4_per_pixel, mapping_num, input_name, output_name):
    base_header(file,scale)
    file.write(f'//!DESC mapping {mapping_num}\n')
    file.write(f'//!COMPUTE {vec4_per_pixel*thread_count} {thread_count} {thread_count} {thread_count}\n')
    file.write(f'//!BIND {input_name}\n')
    file.write(f'//!SAVE {output_name}\n')
    file.write(f'//!COMPONENTS 4\n')
    file.write(f'//!WIDTH LUMA.w {vec4_per_pixel} *\n')
    file.write(f'//!HEIGHT LUMA.h\n')

def header_subband(file, scale, thread_count, vec4_per_pixel, mapping_input_name):
    base_header(file,scale)
    file.write(f'//!DESC sub-band residuals\n')
    file.write(f'//!COMPUTE {vec4_per_pixel*thread_count} {thread_count} {thread_count} {thread_count}\n')
    file.write(f'//!BIND FEATURE\n')
    file.write(f'//!BIND {mapping_input_name}\n')
    file.write(f'//!SAVE RES\n')
    file.write(f'//!COMPONENTS 4\n')
    file.write(f'//!WIDTH LUMA.w {vec4_per_pixel} *\n')
    file.write(f'//!HEIGHT LUMA.h\n')

def header_subconv(file, scale, thread_count, vec4_per_pixel):
    base_header(file,scale)
    file.write(f'//!DESC sub-pixel convolution\n')
    file.write(f'//!COMPUTE {vec4_per_pixel*thread_count} {thread_count} {thread_count} {thread_count}\n')
    file.write(f'//!BIND RES\n')
    file.write(f'//!SAVE SUBCONV\n')
    file.write(f'//!COMPONENTS 4\n')
    file.write(f'//!WIDTH LUMA.w {vec4_per_pixel} *\n')
    file.write(f'//!HEIGHT LUMA.h\n')

def header_aggregation(file, scale, thread_count):
    base_header(file,scale)
    file.write(f'//!DESC aggregation\n')
    file.write(f'//!WIDTH LUMA.w {scale} *\n')
    file.write(f'//!HEIGHT LUMA.h {scale} *\n')
    file.write(f'//!COMPUTE {2*thread_count} {2*thread_count} {thread_count} {thread_count}\n')
    file.write(f'//!BIND SUBCONV\n')




def main():
  if len(sys.argv) == 2:
    fpath=Path(sys.argv[1])
    filename = fpath.stem
    thread_count = 32
    scale, d, s, m, r = [int(i) for i in filename.split(".")[0].split("_")[1:6]]
    if s != 0 or r != 1 or d%4 != 0 or scale != 2:
        print("Unsupported. Only compatible with x2 scale, x-0-x-1 radius 1 models.")
        sys.exit(1)

    dst = fpath.parent / Path(filename.replace("_", "-").replace(filename.split("_")[0], f"FSRCNNX-x{scale}") + ".fastv2.glsl")
    with open(dst, 'w') as file:
        if True:
            # Feature layer
            print("Feature layer: ")

            weight_num = 1
            vec4_per_pixel = d//4
            conv_radius = 2
            conv_width = (conv_radius*2+1)
            conv_surface = conv_width**2
            weights = read_weights(fpath,f"w{weight_num}", conv_surface)
            biases = read_weights(fpath,f"b{weight_num}")
            file.write(f'//w{weight_num} b{weight_num}\n')
            header_feature(file, scale, thread_count, vec4_per_pixel)
            file.write('void hook()\n')
            file.write('{\n')
            file.write(f'const ivec2 sizePerThread = ivec2({vec4_per_pixel},1);\n')
            file.write(f'const ivec2 local_pixel = ivec2(gl_GlobalInvocationID)*sizePerThread;\n')
            file.write(f'const vec2 pos = local_pixel/vec2({vec4_per_pixel},1)*LUMA_pt;\n')
            file.write(gather_5x5.replace("HOOKED","LUMA"))




            for n in range(vec4_per_pixel):
                file.write(f'vec4 res{n} = vec4({format_weights(biases[0], n*4)});\n')


            for conv_num in range(conv_surface):
                weight_num = conv_num*1
                y, x = conv_num%conv_width-conv_radius, conv_num//conv_width-conv_radius
                assert((x >= -conv_radius or x <= conv_radius) and (y >= -conv_radius or y <= conv_radius))
                for n in range(vec4_per_pixel):
                    file.write(f'res{n} += vec4({format_weights(weights[weight_num], n*4)}) * pixel[{x+conv_radius}][{y+conv_radius}];\n')


            for n in range(vec4_per_pixel):
                file.write(f'imageStore(out_image, local_pixel+ivec2({n},0), res{n});\n')

            file.write('}\n\n')


        last_mapping_tex = None
        if True:
            # Mapping layers
            print("Mapping layers: ")
            next_tex = ""
            for mapping_num in range(m):
                previous_tex = "FEATURE" if mapping_num == 0 else f"MAPPING{mapping_num-1}"
                next_tex = f"MAPPING{mapping_num}"

                layer_num = 3+mapping_num
                vec4_per_pixel = d//4
                conv_radius = 1
                conv_width = (conv_radius*2+1)
                conv_surface = conv_width**2
                alpha_num = layer_num+1 #if mapping_num != m-1 else 2 #add relu from next layer at end of first, except for the last which takes the relu that is post-mapping block
                weights = read_weights(fpath,f"w{layer_num}", conv_surface*d)
                biases = read_weights(fpath,f"b{layer_num}")
                alphas = read_weights(fpath,f"alpha{alpha_num}")
                file.write(f'//w{layer_num} b{layer_num} alpha{alpha_num}\n')

                header_mapping(file, scale, thread_count, vec4_per_pixel, mapping_num, previous_tex, next_tex)
                file.write('void hook()\n')
                file.write('{\n')
                for n in range(vec4_per_pixel):
                    file.write(f'vec4 res{n} = vec4({format_weights(biases[0], n*4)});\n')

                file.write(f'const float vec4_per_pixel = {vec4_per_pixel};\n')
                file.write(f'const ivec2 sizePerThread = ivec2({vec4_per_pixel},1);\n')
                file.write(f'const ivec2 local_pixel = ivec2(gl_GlobalInvocationID)*sizePerThread;\n')
                file.write(f'const vec2 samplePos = (local_pixel+vec2(0.5)) * {previous_tex}_pt;\n')



                for conv_num in range(conv_surface):
                    weight_num = conv_num*d
                    y, x = conv_num%conv_width-conv_radius, conv_num//conv_width-conv_radius
                    for prev_n in range(vec4_per_pixel): #previous layer depth dimension
                        prev_n_offset = prev_n*4
                        for cur_n in range(vec4_per_pixel): #current layer depth dimension
                            assert((x >= -conv_radius or x <= conv_radius) and (y >= -conv_radius or y <= conv_radius))
                            print(f'weight_num:{weight_num} prev_n:{prev_n} cur_n:{cur_n} prev_n_offset:{prev_n_offset} weight_num+prev_n_offset+0:{weight_num+prev_n_offset+0} len(weights):{len(weights)} x:{x} y:{y}')
                            file.write('res{} += mat4({},{},{},{}) '.format(cur_n,
                                                                            format_weights(weights[weight_num+prev_n_offset+0], cur_n*4),
                                                                            format_weights(weights[weight_num+prev_n_offset+1], cur_n*4),
                                                                            format_weights(weights[weight_num+prev_n_offset+2], cur_n*4),
                                                                            format_weights(weights[weight_num+prev_n_offset+3], cur_n*4)) + \
                                                                            f'* {previous_tex}_tex(samplePos + (vec2({x},{y})*vec2(vec4_per_pixel,1) + vec2({prev_n},{0})) * {previous_tex}_pt);\n')


                for n in range(vec4_per_pixel):
                    # if mapping_num > 0: #relu is from beginning of next layer
                    file.write(f'res{n} = max(res{n}, vec4(0.0)) + vec4({format_weights(alphas[0], n*4)}) * min(res{n}, vec4(0.0));\n')
                    file.write(f'imageStore(out_image, local_pixel+ivec2({n},0), res{n});\n')

                file.write('}\n\n')
            last_mapping_tex = next_tex

        if True:
            #Subband layer
            print("Subband layer : ")
            layer_num = 3+m
            vec4_per_pixel = d//4
            conv_width = 1
            conv_surface = conv_width**2
            file.write(f'//w{layer_num} b{layer_num} alpha2\n')
            weights = read_weights(fpath,f"w{layer_num}", d)
            biases = read_weights(fpath,f"b{layer_num}")
            alphas = read_weights(fpath,f"alpha2")

            header_subband(file, scale, thread_count, vec4_per_pixel, last_mapping_tex)
            file.write('void hook()\n')
            file.write('{\n')
            for n in range(vec4_per_pixel):
                file.write(f'vec4 res{n} = vec4({format_weights(biases[0], n*4)});\n')

            file.write(f'const float vec4_per_pixel = {vec4_per_pixel};\n')
            file.write(f'const ivec2 sizePerThread = ivec2({vec4_per_pixel},1);\n')
            file.write(f'const ivec2 local_pixel = ivec2(gl_GlobalInvocationID)*sizePerThread;\n')
            file.write(f'const vec2 samplePos = (local_pixel+vec2(0.5)) * {last_mapping_tex}_pt;\n')


            for prev_n in range(vec4_per_pixel): #previous layer depth dimension
                prev_n_offset = prev_n*4
                for cur_n in range(vec4_per_pixel): #current layer depth dimension
                    print(f'prev_n:{prev_n} cur_n:{cur_n} prev_n_offset:{prev_n_offset} prev_n_offset+0:{prev_n_offset+0} prev_n_offset+3:{prev_n_offset+3} len(weights):{len(weights)} x:{prev_n_offset} y:0')
                    file.write('res{} += mat4({},{},{},{}) '.format(cur_n,
                                                                    format_weights(weights[prev_n_offset+0], cur_n*4),
                                                                    format_weights(weights[prev_n_offset+1], cur_n*4),
                                                                    format_weights(weights[prev_n_offset+2], cur_n*4),
                                                                    format_weights(weights[prev_n_offset+3], cur_n*4)) + \
                                                                    f'* {last_mapping_tex}_tex(samplePos + (vec2({prev_n},{0})) * {last_mapping_tex}_pt);\n')

            for n in range(vec4_per_pixel):
                file.write(f'res{n} += FEATURE_tex(samplePos + (vec2({n},0)) * MAPPING{m-1}_pt);\n')
                file.write(f'res{n} = max(res{n}, vec4(0.0)) + vec4({format_weights(alphas[0], n*4)}) * min(res{n}, vec4(0.0));\n')
                file.write(f'imageStore(out_image, local_pixel+ivec2({n},0), res{n});\n')

            file.write('}\n\n')



        if True:
            #Sub-pixel convolution
            print("Subpixel conv layer : ")
            previous_tex = "RES"
            d_vec4_per_pixel = d//4
            sub_layer_count = scale**2
            sub_vec4_per_pixel = sub_layer_count//4
            conv_radius = 1
            conv_width = (conv_radius*2+1)
            conv_surface = conv_width**2
            weights = read_weights(fpath,f"deconv_w", conv_surface*d)
            biases = read_weights(fpath,f"deconv_b")
            file.write(f'//deconv_w deconv_b\n')

            header_subconv(file, scale, thread_count, sub_vec4_per_pixel)
            file.write('void hook()\n')
            file.write('{\n')
            for n in range(sub_vec4_per_pixel):
                file.write(f'vec4 res{n} = vec4({format_weights(biases[0], n*4)});\n')

            file.write(f'const float cur_vec4_per_pixel = {sub_vec4_per_pixel};\n')
            file.write(f'const float source_vec4_per_pixel = {d_vec4_per_pixel};\n')
            file.write(f'const ivec2 sizePerThread = ivec2({sub_vec4_per_pixel},1);\n')
            file.write(f'const ivec2 local_pixel = ivec2(gl_GlobalInvocationID)*sizePerThread;\n')
            file.write(f'const vec2 real_pixel = local_pixel/vec2(cur_vec4_per_pixel,1);\n')
            file.write(f'const vec2 samplePos = (vec2(real_pixel)*vec2(source_vec4_per_pixel,1)+vec2(0.5)) * {previous_tex}_pt;\n')

            for conv_num in range(conv_surface):
                weight_num = conv_num*d
                y, x = conv_num%conv_width-conv_radius, conv_num//conv_width-conv_radius
                for prev_n in range(d_vec4_per_pixel): #previous layer depth dimension
                    prev_n_offset = prev_n*4
                    for cur_n in range(sub_vec4_per_pixel): #current layer depth dimension
                        print(f'weight_num:{weight_num} prev_n:{prev_n} cur_n:{cur_n} prev_n_offset:{prev_n_offset} weight_num+prev_n_offset+0:{weight_num+prev_n_offset+0} len(weights):{len(weights)} x:{x} y:{y}')
                        assert((x >= -conv_radius or x <= conv_radius) and (y >= -conv_radius or y <= conv_radius))
                        file.write('res{} += mat4({},{},{},{}) '.format(cur_n,
                                                                        format_weights(weights[weight_num+prev_n_offset+0], cur_n*4),
                                                                        format_weights(weights[weight_num+prev_n_offset+1], cur_n*4),
                                                                        format_weights(weights[weight_num+prev_n_offset+2], cur_n*4),
                                                                        format_weights(weights[weight_num+prev_n_offset+3], cur_n*4)) + \
                                                                        f'* {previous_tex}_tex(samplePos + (vec2({x},{y})*vec2(source_vec4_per_pixel,1) + vec2({prev_n},{0})) * {previous_tex}_pt);\n')


            for n in range(sub_vec4_per_pixel):
                file.write(f'imageStore(out_image, local_pixel+ivec2({n},0), res{n});\n')

            file.write('}\n\n')



        #TESTED GOOD
        if True:
            #Depth to space
            print("Depth to space layer : ")
            if scale == 2:
                #FAST x2
                header_aggregation(file,scale, thread_count)

                file.write('void hook()\n')
                file.write('{\n')
                file.write(f'const ivec2 real_pixel = ivec2(gl_GlobalInvocationID);\n')
                file.write(f'const ivec2 local_pixel = real_pixel*ivec2(2,2);\n')
                file.write(f'const vec2 samplePos = (vec2(real_pixel)+vec2(0.5)) * SUBCONV_pt;\n')

                file.write(f'const vec4 value = vec4(SUBCONV_tex(samplePos));\n')
                file.write(f'imageStore(out_image, local_pixel+ivec2(0,0), vec4(value.x,0,0,1));\n')
                file.write(f'imageStore(out_image, local_pixel+ivec2(0,1), vec4(value.y,0,0,1));\n')
                file.write(f'imageStore(out_image, local_pixel+ivec2(1,0), vec4(value.z,0,0,1));\n')
                file.write(f'imageStore(out_image, local_pixel+ivec2(1,1), vec4(value.w,0,0,1));\n')
                file.write('}\n')

            else:
                print("BROKEN, needs header need to be modified. This is not compute compatible.")
                sys.exit(1)
                # sub_vec4_per_pixel = (scale**2)/4#4
                # # Aggregation
                # header_aggregation(file, scale)
                # file.write('vec4 hook()\n')
                # file.write('{\n')
                # file.write(f'vec2 fcoord = fract(SUBCONV_pos * SUBCONV_size/{sub_vec4_per_pixel} );\n') #//pos in pix in input looped to 1, meaning sub pix offset from pixel in input
                # file.write(f'vec2 base = SUBCONV_pos + vec2(0.5)* SUBCONV_pt - (fcoord * SUBCONV_pt  * {sub_vec4_per_pixel});\n') #//position at exact center of pixel in input (offset removed)
                # file.write(f'ivec2 index = ivec2(fcoord * vec2({scale}));\n')  #//offset in output pixels from the center of origin pixel
                # file.write(f'int index_layer = index.x * {scale} + index.y;\n')
                # file.write(f'int index_id = index_layer%4;\n')
                # file.write(f'int index_offset = index_layer/4;\n')
                # file.write(f'vec2 texpos = base+vec2(index_offset*SUBCONV_pt.x,0);\n')
                # file.write(f'vec4 res = SUBCONV_tex(texpos);\n')
                # file.write(f'return vec4(res[index_id], 0, 0, 1);\n')
                # file.write('}\n')

    print(f'Saved "{filename}" to "{dst}".')
  else:
    print("Missing argument: You must specify a file name.")
    return

if __name__ == '__main__':
  main()
