import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Outputs all subfigures given the predefined sigmas",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-bar', '--bar', action='store_true',
                        help='Changes the plot type from EAC to bar')  
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    models = ['ClaudesLens_Logistic', 'ClaudesLens_ConvNext', 'ClaudesLens_ViT', 'Pretrained_ViT_B_16', 'Pretrained_ConvNext']
    noise_types = ['image', 'weight']
    sigmas = ['0.000', '0.100', '0.500', '1.000', '10.000']
    start = '\\begin{figure}[H]\n' + '  \\centering\n'
    end   = '\\end{figure}\n\n'
    subfig_end = '  \\end{subfigure}\n'
    plot_types = ['Barplot', 'EAC']
    plot_dirs = ['/bar_plots/', '/EAC/']
    file_names = ['bar_', 'eac_']

    file_name = file_names[0] if args.bar else file_names[1] 
    plot_type = plot_types[0] if args.bar else plot_types[1]
    plot_dir  = plot_dirs[0]  if args.bar else plot_dirs[1]

    
    
    for model in models:
        subpath = 'imgs/' + model + plot_dir 
    
        for noise_type in noise_types:
            path = subpath + noise_type
            outputs = []
            subfig = '  \\begin{subfigure}[b]{0.4\\textwidth}\n'
            subfig += '    \\centering\n' + '    \\includegraphics[width=\\textwidth]{'
            
    
            for sigma in sigmas:
                caption_type = plot_type
                label_type   = plot_type.lower()
                
                temp       = subfig + f'{path}/{file_name}{sigma}.png}}\n'
                model_temp = model.replace('ClaudesLens', 'modified')
                model_temp = model_temp.replace('_B_16', '')
                
                temp += f'    \\caption{{{caption_type} with {noise_type} perturbation $\\sigma = {sigma}$}}\n'
                temp += f'    \\label{{fig:{model_temp}_{label_type}_{noise_type}_{sigma}}}\n'.replace('.','_').lower()
    
                outputs.append(temp)
    
        
            out = start
            for idx, output in enumerate(outputs):
                if idx % 2 == 0:
                    hfill = '  \\hfill\n'
                else: hfill = '\n'
                out += output + subfig_end + hfill
            
            output = out + end
    
            with open(f"{model}.tex", "a") as f:
                f.write(output)
    

main()



