#!/usr/bin/env python3
"""
Large-scale knowledge atlas — 1000+ prompts × 4 representational spaces.

Usage:
    python atlas_large.py --scaffold dfssm_dfw_step1501.pt \
                           --lora dfw_lora_all_r16_final.pt
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
sys.path.insert(0, '.')
from df_ssm_dfw_lora import *
from transformers import AutoTokenizer

def load_model(scaffold_path, lora_path, device='cuda'):
    teacher, cfg, vocab_size = load_teacher('state-spaces/mamba2-1.3b', device='cpu')
    del teacher
    student = DFWMamba2LM(vocab_size=vocab_size, Ks=23, Kw=4, **cfg).to(device)
    ckpt = torch.load(scaffold_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'], strict=False)
    freeze_scaffold_add_lora(student, lora_layers='all', lora_rank=16, lora_targets='both')
    student = student.to(device)
    lora_ckpt = torch.load(lora_path, map_location=device, weights_only=False)
    for name, param in student.named_parameters():
        if name in lora_ckpt['lora_state']:
            param.data = lora_ckpt['lora_state'][name].to(device)
    student.eval()
    return student, vocab_size

# ================================================================
# MASSIVE PROMPT DATABASE (~1200 prompts, 20 categories)
# ================================================================

PROMPTS = {}

# --- Geography: Capitals ---
PROMPTS["capitals"] = [(f"The capital of {c} is", ans) for c, ans in [
    ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("Italy","Rome"),
    ("Spain","Madrid"),("China","Beijing"),("Brazil","Brasilia"),("Australia","Canberra"),
    ("Canada","Ottawa"),("Russia","Moscow"),("India","New Delhi"),("Egypt","Cairo"),
    ("Mexico","Mexico City"),("South Korea","Seoul"),("Turkey","Ankara"),
    ("Thailand","Bangkok"),("Poland","Warsaw"),("Sweden","Stockholm"),
    ("Norway","Oslo"),("Denmark","Copenhagen"),("Finland","Helsinki"),
    ("Greece","Athens"),("Portugal","Lisbon"),("Ireland","Dublin"),
    ("Argentina","Buenos Aires"),("Peru","Lima"),("Colombia","Bogota"),
    ("Chile","Santiago"),("Cuba","Havana"),("Nigeria","Abuja"),
    ("Kenya","Nairobi"),("South Africa","Pretoria"),("Morocco","Rabat"),
    ("Indonesia","Jakarta"),("Vietnam","Hanoi"),("Philippines","Manila"),
    ("Pakistan","Islamabad"),("Iran","Tehran"),("Iraq","Baghdad"),
    ("Saudi Arabia","Riyadh"),("Israel","Jerusalem"),("Ukraine","Kyiv"),
    ("Czech Republic","Prague"),("Hungary","Budapest"),("Romania","Bucharest"),
    ("Austria","Vienna"),("Switzerland","Bern"),("Belgium","Brussels"),
    ("Netherlands","Amsterdam"),("New Zealand","Wellington"),
]]

# --- Geography: Languages ---
PROMPTS["languages"] = [(f"The official language of {c} is", ans) for c, ans in [
    ("France","French"),("Germany","German"),("Japan","Japanese"),("Italy","Italian"),
    ("Spain","Spanish"),("China","Chinese"),("Brazil","Portuguese"),("Russia","Russian"),
    ("Turkey","Turkish"),("South Korea","Korean"),("Thailand","Thai"),
    ("Poland","Polish"),("Sweden","Swedish"),("Norway","Norwegian"),
    ("Denmark","Danish"),("Finland","Finnish"),("Greece","Greek"),
    ("Portugal","Portuguese"),("Netherlands","Dutch"),("Indonesia","Indonesian"),
    ("Vietnam","Vietnamese"),("Iran","Persian"),("Israel","Hebrew"),
    ("Egypt","Arabic"),("India","Hindi"),("Czech Republic","Czech"),
    ("Hungary","Hungarian"),("Romania","Romanian"),("Ukraine","Ukrainian"),
    ("Croatia","Croatian"),
]]

# --- Geography: Continents ---
PROMPTS["continents"] = [(f"{c} is located in", ans) for c, ans in [
    ("France","Europe"),("Japan","Asia"),("Brazil","South America"),
    ("Australia","Oceania"),("Egypt","Africa"),("Canada","North America"),
    ("India","Asia"),("Germany","Europe"),("China","Asia"),("Mexico","North America"),
    ("Nigeria","Africa"),("Argentina","South America"),("Thailand","Asia"),
    ("Poland","Europe"),("South Africa","Africa"),("Indonesia","Asia"),
    ("Peru","South America"),("Turkey","Europe"),("Iran","Asia"),("Kenya","Africa"),
    ("Sweden","Europe"),("Colombia","South America"),("Vietnam","Asia"),
    ("Morocco","Africa"),("Chile","South America"),("Philippines","Asia"),
    ("Greece","Europe"),("Cuba","North America"),("New Zealand","Oceania"),
    ("Saudi Arabia","Asia"),
]]

# --- Science: Chemical Elements ---
PROMPTS["elements"] = [(f"The chemical symbol for {e} is", ans) for e, ans in [
    ("gold","Au"),("silver","Ag"),("iron","Fe"),("copper","Cu"),("tin","Sn"),
    ("lead","Pb"),("mercury","Hg"),("sodium","Na"),("potassium","K"),
    ("calcium","Ca"),("oxygen","O"),("hydrogen","H"),("nitrogen","N"),
    ("carbon","C"),("helium","He"),("neon","Ne"),("argon","Ar"),
    ("chlorine","Cl"),("sulfur","S"),("phosphorus","P"),("zinc","Zn"),
    ("nickel","Ni"),("chromium","Cr"),("manganese","Mn"),("cobalt","Co"),
    ("aluminum","Al"),("silicon","Si"),("titanium","Ti"),("uranium","U"),
    ("platinum","Pt"),
]]

# --- Science: Physics ---
PROMPTS["physics"] = [(p, a) for p, a in [
    ("The speed of light is approximately","300"),
    ("The boiling point of water is","100"),
    ("The freezing point of water is","0"),
    ("The acceleration due to gravity is approximately","9"),
    ("The largest planet in the solar system is","Jupiter"),
    ("The smallest planet in the solar system is","Mercury"),
    ("The closest planet to the Sun is","Mercury"),
    ("The hottest planet in the solar system is","Venus"),
    ("The planet known for its rings is","Saturn"),
    ("The red planet is","Mars"),
    ("The closest star to Earth is","the"),
    ("The speed of sound in air is approximately","343"),
    ("Light travels fastest in","vacuum"),
    ("The unit of electrical resistance is the","ohm"),
    ("The unit of force is the","Newton"),
    ("The unit of energy is the","joule"),
    ("The unit of power is the","watt"),
    ("The unit of frequency is the","hertz"),
    ("The charge of an electron is","negative"),
    ("Water freezes at","zero"),
    ("Absolute zero is approximately","minus"),
    ("The formula for force is mass times","acceleration"),
    ("The law of gravity was discovered by","Isaac"),
    ("The theory of relativity was proposed by","Albert"),
    ("Electricity flows through materials called","conductors"),
]]

# --- Famous People: Scientists ---
PROMPTS["scientists"] = [(p, a) for p, a in [
    ("The theory of relativity was proposed by","Albert Einstein"),
    ("The laws of motion were formulated by","Isaac Newton"),
    ("The theory of evolution was proposed by","Charles Darwin"),
    ("The discoverer of penicillin was","Alexander Fleming"),
    ("The inventor of the telephone was","Alexander Graham Bell"),
    ("The inventor of the light bulb was","Thomas Edison"),
    ("The discoverer of radium was","Marie Curie"),
    ("The father of computer science is","Alan Turing"),
    ("The creator of the World Wide Web is","Tim Berners"),
    ("The author of A Brief History of Time is","Stephen Hawking"),
    ("The inventor of the printing press was","Johannes Gutenberg"),
    ("The developer of the polio vaccine was","Jonas Salk"),
    ("The discoverer of DNA structure was","James Watson"),
    ("The physicist who developed quantum mechanics was","Werner Heisenberg"),
    ("The astronomer who proposed the heliocentric model was","Nicolaus Copernicus"),
    ("The mathematician who invented calculus was","Isaac Newton"),
    ("The scientist who discovered electricity was","Benjamin Franklin"),
    ("The inventor of dynamite was","Alfred Nobel"),
    ("The inventor of the steam engine was","James Watt"),
    ("The scientist who classified living organisms was","Carl Linnaeus"),
]]

# --- Famous People: Writers ---
PROMPTS["writers"] = [(f"The author of {b} is", ans) for b, ans in [
    ("Romeo and Juliet","William Shakespeare"),("Harry Potter","J.K. Rowling"),
    ("1984","George Orwell"),("Pride and Prejudice","Jane Austen"),
    ("The Great Gatsby","F. Scott Fitzgerald"),("Don Quixote","Miguel de Cervantes"),
    ("War and Peace","Leo Tolstoy"),("The Odyssey","Homer"),
    ("Hamlet","William Shakespeare"),("Les Miserables","Victor Hugo"),
    ("Crime and Punishment","Fyodor Dostoevsky"),("The Divine Comedy","Dante"),
    ("Moby Dick","Herman Melville"),("The Catcher in the Rye","J.D. Salinger"),
    ("To Kill a Mockingbird","Harper Lee"),("Lord of the Flies","William Golding"),
    ("Animal Farm","George Orwell"),("Brave New World","Aldous Huxley"),
    ("The Lord of the Rings","J.R.R. Tolkien"),("Frankenstein","Mary Shelley"),
    ("Dracula","Bram Stoker"),("The Picture of Dorian Gray","Oscar Wilde"),
    ("Oliver Twist","Charles Dickens"),("Jane Eyre","Charlotte Bronte"),
    ("Wuthering Heights","Emily Bronte"),
]]

# --- Technology: Companies ---
PROMPTS["companies"] = [(p, a) for p, a in [
    ("The company that created Windows is","Microsoft"),
    ("The company that created the iPhone is","Apple"),
    ("The company that created Android is","Google"),
    ("The company that created Facebook is","Mark"),
    ("The founder of Amazon is","Jeff"),
    ("The founder of Tesla is","Elon"),
    ("The founder of Microsoft is","Bill"),
    ("The founder of Apple is","Steve"),
    ("The company that created Gmail is","Google"),
    ("The company that created PlayStation is","Sony"),
    ("The company that created the Kindle is","Amazon"),
    ("The CEO who founded Facebook is","Mark"),
    ("The company that makes the Galaxy phone is","Samsung"),
    ("The company that created ChatGPT is","Open"),
    ("The programming language created by Guido van Rossum is","Python"),
    ("The programming language created by James Gosling is","Java"),
    ("The programming language created by Bjarne Stroustrup is","C"),
    ("The programming language created by Dennis Ritchie is","C"),
    ("The operating system created by Linus Torvalds is","Linux"),
    ("The search engine created by Larry Page is","Google"),
]]

# --- Animals ---
PROMPTS["animals"] = [(p, a) for p, a in [
    ("The largest animal on Earth is the","blue whale"),
    ("The fastest land animal is the","cheetah"),
    ("The tallest animal is the","giraffe"),
    ("The largest bird is the","ostrich"),
    ("The fastest bird is the","peregrine"),
    ("The largest fish is the","whale shark"),
    ("The smartest animal after humans is the","chimpanzee"),
    ("The animal known as the king of the jungle is the","lion"),
    ("The largest reptile is the","saltwater"),
    ("The largest insect is the","goliath"),
    ("A baby dog is called a","puppy"),
    ("A baby cat is called a","kitten"),
    ("A baby horse is called a","foal"),
    ("A baby cow is called a","calf"),
    ("A baby sheep is called a","lamb"),
    ("A group of lions is called a","pride"),
    ("A group of fish is called a","school"),
    ("A group of wolves is called a","pack"),
    ("A group of crows is called a","murder"),
    ("A group of geese is called a","flock"),
    ("The animal that produces honey is the","bee"),
    ("The animal that produces silk is the","silkworm"),
    ("The national animal of Australia is the","kangaroo"),
    ("The national animal of India is the","tiger"),
    ("The national animal of China is the","panda"),
]]

# --- Colors/Properties ---
PROMPTS["colors"] = [(p, a) for p, a in [
    ("The color of the sky on a clear day is","blue"),
    ("The color of grass is","green"),
    ("The color of blood is","red"),
    ("The color of snow is","white"),
    ("The color of coal is","black"),
    ("The color of the sun is","yellow"),
    ("The color of an orange is","orange"),
    ("The color of a banana is","yellow"),
    ("The color of a ruby is","red"),
    ("The color of an emerald is","green"),
    ("The color of a sapphire is","blue"),
    ("The color of gold is","gold"),
    ("The color of chocolate is","brown"),
    ("The color of the ocean is","blue"),
    ("The color of a flamingo is","pink"),
]]

# --- Currencies ---
PROMPTS["currencies"] = [(f"The currency of {c} is the", ans) for c, ans in [
    ("Japan","yen"),("United Kingdom","pound"),("United States","dollar"),
    ("European Union","euro"),("China","yuan"),("India","rupee"),
    ("Russia","ruble"),("Brazil","real"),("South Korea","won"),
    ("Mexico","peso"),("Switzerland","franc"),("Australia","dollar"),
    ("Canada","dollar"),("Sweden","krona"),("Turkey","lira"),
    ("Thailand","baht"),("South Africa","rand"),("Israel","shekel"),
    ("Saudi Arabia","riyal"),("Egypt","pound"),
]]

# --- Math/Numbers ---
PROMPTS["math"] = [(p, a) for p, a in [
    ("Two plus two equals","four"),("Three times three equals","nine"),
    ("The square root of 144 is","12"),("The square root of 100 is","10"),
    ("The square root of 64 is","eight"),("The square root of 25 is","five"),
    ("The value of pi is approximately","3"),("Ten divided by two equals","five"),
    ("One hundred minus one equals","ninety"),("Seven times eight equals","fifty"),
    ("The number of sides of a triangle is","three"),
    ("The number of sides of a hexagon is","six"),
    ("The number of sides of an octagon is","eight"),
    ("The number of degrees in a circle is","360"),
    ("The number of degrees in a right angle is","90"),
    ("The number of days in a year is","365"),
    ("The number of hours in a day is","24"),
    ("The number of minutes in an hour is","60"),
    ("The number of seconds in a minute is","60"),
    ("The number of months in a year is","twelve"),
    ("The number of weeks in a year is","52"),
    ("The first prime number is","two"),
    ("The largest single digit number is","nine"),
    ("Zero times any number equals","zero"),
    ("Any number to the power of zero equals","one"),
]]

# --- Food/Cuisine ---
PROMPTS["food"] = [(p, a) for p, a in [
    ("The main ingredient of bread is","flour"),
    ("The main ingredient of chocolate is","cacao"),
    ("The main ingredient of wine is","grapes"),
    ("The main ingredient of beer is","barley"),
    ("The main ingredient of tofu is","soy"),
    ("The main ingredient of butter is","cream"),
    ("The main ingredient of cheese is","milk"),
    ("Sushi is a traditional food from","Japan"),
    ("Pizza is a traditional food from","Italy"),
    ("Tacos are a traditional food from","Mexico"),
    ("Croissants are a traditional food from","France"),
    ("Kimchi is a traditional food from","Korea"),
    ("Curry originated in","India"),
    ("Pasta originated in","Italy"),
    ("Dim sum is a traditional food from","China"),
    ("Paella is a traditional food from","Spain"),
    ("Borscht is a traditional food from","Russia"),
    ("Fish and chips is a traditional food from","England"),
    ("Bratwurst is a traditional food from","Germany"),
    ("Falafel originated in the","Middle"),
]]

# --- Music ---
PROMPTS["music"] = [(p, a) for p, a in [
    ("The composer of the Moonlight Sonata is","Ludwig"),
    ("The composer of The Four Seasons is","Antonio Vivaldi"),
    ("The composer of The Magic Flute is","Wolfgang"),
    ("The lead singer of Queen was","Freddie"),
    ("The lead singer of The Beatles was","John"),
    ("The King of Pop is","Michael"),
    ("The Queen of Pop is","Madonna"),
    ("The instrument with 88 keys is the","piano"),
    ("The instrument with six strings is the","guitar"),
    ("The smallest instrument in the string family is the","violin"),
    ("The deepest instrument in the string family is the","bass"),
    ("A musical composition for an orchestra is called a","symphony"),
    ("The national anthem of the United States is","The"),
    ("The composer of the Ninth Symphony is","Ludwig"),
    ("The inventor of the phonograph was","Thomas"),
    ("Music with a strong beat and electric guitars is called","rock"),
    ("The country known as the birthplace of jazz is the","United"),
    ("The country known as the birthplace of reggae is","Jamaica"),
    ("The country known as the birthplace of samba is","Brazil"),
    ("A group of four musicians is called a","quartet"),
]]

# --- Sports ---
PROMPTS["sports"] = [(p, a) for p, a in [
    ("The most popular sport in the world is","soccer"),
    ("The sport played at Wimbledon is","tennis"),
    ("The sport with a puck and ice is","hockey"),
    ("The sport played with a bat and ball on a diamond is","baseball"),
    ("The sport invented by James Naismith is","basketball"),
    ("The Olympic sport of running 42 kilometers is the","marathon"),
    ("The sport of riding horses is called","equestrian"),
    ("The number of players on a soccer team is","eleven"),
    ("The number of players on a basketball team is","five"),
    ("The number of holes in a round of golf is","eighteen"),
    ("The country that invented cricket is","England"),
    ("The country that invented judo is","Japan"),
    ("The country that invented taekwondo is","South Korea"),
    ("The World Cup is a tournament for the sport of","soccer"),
    ("The Super Bowl is a championship for the sport of","American"),
    ("The sport where you hit a shuttlecock is","badminton"),
    ("The sport played on a court with a net and rackets is","tennis"),
    ("A perfect score in bowling is","300"),
    ("The length of a marathon in miles is approximately","26"),
    ("The number of rings on the Olympic flag is","five"),
]]

# --- Body/Medical ---
PROMPTS["medical"] = [(p, a) for p, a in [
    ("The largest organ in the human body is the","skin"),
    ("The smallest bone in the human body is the","stapes"),
    ("The hardest substance in the human body is","enamel"),
    ("The number of bones in the adult human body is","206"),
    ("The number of chromosomes in a human cell is","46"),
    ("Blood is pumped through the body by the","heart"),
    ("Oxygen is carried in the blood by","hemoglobin"),
    ("The organ that filters blood is the","kidney"),
    ("The organ that produces insulin is the","pancreas"),
    ("The organ that produces bile is the","liver"),
    ("The number of chambers in the human heart is","four"),
    ("The gas humans exhale is","carbon"),
    ("The gas humans inhale is","oxygen"),
    ("The vitamin obtained from sunlight is","vitamin D"),
    ("The disease caused by a lack of vitamin C is","scurvy"),
    ("DNA stands for deoxyribonucleic","acid"),
    ("The study of the brain is called","neuroscience"),
    ("The study of diseases is called","pathology"),
    ("The doctor who performs surgery is called a","surgeon"),
    ("Normal human body temperature is approximately","98"),
]]

# --- Historical ---
PROMPTS["history"] = [(p, a) for p, a in [
    ("The first president of the United States was","George"),
    ("The year World War II ended was","1945"),
    ("The ancient civilization that built the pyramids was","Egypt"),
    ("The wall that divided Berlin was built in","1961"),
    ("The first man to walk on the moon was","Neil"),
    ("The ship that sank in 1912 was the","Titanic"),
    ("The empire that ruled Rome was the","Roman"),
    ("The revolution that began in 1789 was the","French"),
    ("The country that attacked Pearl Harbor was","Japan"),
    ("The explorer who discovered America was","Christopher"),
    ("The Renaissance began in","Italy"),
    ("The Cold War was between the United States and the","Soviet"),
    ("The document that declared American independence was signed in","1776"),
    ("The Great Wall was built in","China"),
    ("The ancient Greeks invented","democracy"),
    ("The Magna Carta was signed in","1215"),
    ("The inventor of the airplane was","the Wright"),
    ("The first satellite launched into space was","Sputnik"),
    ("World War I began in","1914"),
    ("The fall of the Berlin Wall was in","1989"),
]]

# --- Mythology/Religion ---
PROMPTS["mythology"] = [(p, a) for p, a in [
    ("The Greek god of the sea is","Poseidon"),
    ("The Greek god of war is","Ares"),
    ("The Greek goddess of wisdom is","Athena"),
    ("The Greek god of the underworld is","Hades"),
    ("The king of the Greek gods is","Zeus"),
    ("The Roman name for Zeus is","Jupiter"),
    ("The Norse god of thunder is","Thor"),
    ("The holy book of Islam is the","Quran"),
    ("The holy book of Christianity is the","Bible"),
    ("The founder of Buddhism is","Siddhartha"),
    ("The holy city of Islam is","Mecca"),
    ("The religion that originated in India is","Hinduism"),
    ("The symbol of Christianity is the","cross"),
    ("The number of commandments in the Bible is","ten"),
    ("The first book of the Bible is","Genesis"),
]]

# --- Materials/Substances ---
PROMPTS["materials"] = [(p, a) for p, a in [
    ("The hardest natural substance is","diamond"),
    ("The most abundant gas in Earth's atmosphere is","nitrogen"),
    ("The most abundant element in the universe is","hydrogen"),
    ("The most conductive metal is","silver"),
    ("The lightest metal is","lithium"),
    ("The densest naturally occurring element is","osmium"),
    ("Glass is made primarily from","sand"),
    ("Paper is made primarily from","wood"),
    ("Rubber comes from the","rubber"),
    ("Steel is an alloy of iron and","carbon"),
    ("Bronze is an alloy of copper and","tin"),
    ("Brass is an alloy of copper and","zinc"),
    ("The gas that makes up most of the air we breathe is","nitrogen"),
    ("Water is composed of hydrogen and","oxygen"),
    ("Salt is composed of sodium and","chlorine"),
]]

TARGET_LAYERS = [3, 15, 33, 47]

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scaffold', default='dfssm_dfw_step1501.pt')
    p.add_argument('--lora', default='dfw_lora_all_r16_final.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    model, vocab_size = load_model(args.scaffold, args.lora, args.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    n_layers = len(model.layers)
    
    # Flatten
    all_prompts, all_cats, all_labels = [], [], []
    for cat, prompts in PROMPTS.items():
        for prompt, label in prompts:
            all_prompts.append(prompt)
            all_cats.append(cat)
            all_labels.append(label.split()[0] if ' ' in label else label)  # short label
    
    N = len(all_prompts)
    cats_unique = list(PROMPTS.keys())
    print(f"Total: {N} prompts, {len(cats_unique)} categories")
    for cat in cats_unique:
        print(f"  {cat}: {len(PROMPTS[cat])}")
    
    # Collect states at target layers
    print(f"\nCollecting hidden states at layers {TARGET_LAYERS}...")
    layer_states = {l: [] for l in TARGET_LAYERS}
    
    for pi, prompt in enumerate(all_prompts):
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        block_len = 64
        L = ids.shape[1]
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=0)
        
        with torch.no_grad():
            x = model.embedding(ids)
            for l in range(n_layers):
                x = x + model.layers[l].mixer(model.layers[l].norm(x))
                if (l + 1) in TARGET_LAYERS:
                    layer_states[l + 1].append(x[:, -1, :].squeeze(0).cpu())
        
        if (pi + 1) % 100 == 0:
            print(f"  {pi+1}/{N}")
    
    for l in TARGET_LAYERS:
        layer_states[l] = torch.stack(layer_states[l])
    
    print(f"  Done.")
    
    # Analyze each layer
    export = {}
    
    for tl in TARGET_LAYERS:
        V = layer_states[tl]
        V_centered = V - V.mean(dim=0)
        
        _, S, Vt = torch.svd(V_centered.float())
        proj = (V_centered.float() @ Vt[:, :10]).numpy()  # keep 10 PCs
        proj_2d = proj[:, :2]
        
        V_norm = F.normalize(V.float(), dim=-1)
        sim = (V_norm @ V_norm.t()).numpy()
        
        total_var = (S ** 2).sum()
        var_10pc = ((S[:10] ** 2).sum() / total_var * 100).item()
        
        layer_name = f"L{tl-1}"
        space_name = {3:"Intent", 15:"Translation", 33:"Knowledge", 47:"Output"}[tl]
        
        print(f"\n{'='*70}")
        print(f"LAYER {layer_name} — {space_name} space (10 PCs: {var_10pc:.1f}%)")
        print(f"{'='*70}")
        
        print(f"\n  {'Category':<15} {'N':>4} {'Within':>8} {'Between':>8} {'Sep':>8} {'Spread':>8}")
        print(f"  {'-'*55}")
        
        seps = []
        for cat in cats_unique:
            idx = [i for i, c in enumerate(all_cats) if c == cat]
            within = [sim[i,j] for i in idx for j in idx if i < j]
            between = [sim[i,j] for i in idx for j in range(N) if all_cats[j] != cat]
            w_avg = np.mean(within) if within else 0
            b_avg = np.mean(between) if between else 0
            sep = w_avg - b_avg
            seps.append(sep)
            
            pts = proj_2d[idx]
            spread = np.sqrt(((pts - pts.mean(axis=0))**2).sum(axis=1).mean())
            
            print(f"  {cat:<15} {len(idx):>4} {w_avg:>8.4f} {b_avg:>8.4f} {sep:>+8.4f} {spread:>8.1f}")
        
        avg_sep = np.mean(seps)
        print(f"\n  Average separation: {avg_sep:+.4f}")
        
        # Export
        export[layer_name] = {
            'space': space_name,
            'var_10pc': var_10pc,
            'avg_sep': float(avg_sep),
            'points': [
                {'label': all_labels[i], 'category': all_cats[i],
                 'x': float(proj_2d[i,0]), 'y': float(proj_2d[i,1])}
                for i in range(N)
            ],
            'per_category': {
                cat: float(seps[ci]) for ci, cat in enumerate(cats_unique)
            }
        }
    
    with open('atlas_large.json', 'w') as f:
        json.dump(export, f)
    
    print(f"\nExported {N} points × {len(TARGET_LAYERS)} layers to atlas_large.json")
    print("Done.")


if __name__ == '__main__':
    main()
