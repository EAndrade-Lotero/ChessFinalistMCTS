import numpy as np

from collections import namedtuple 
from random import sample, choice

Tupla = namedtuple('Tupla', ['elemento', 'valor'])

class ListaPrioritaria():
    
    def __init__(self):
        self.pila = []
        
    def __len__(self):
        return len(self.pila)

    def push(self, elemento, valor):
        tupla = Tupla(elemento, valor)
        self.pila.append(tupla)
        self.pila.sort(key=lambda x: x[1])
            
    def pop(self, with_value=False):
        if with_value:
            return self.pila.pop(0)
        else:
            return self.pila.pop(0)[0]
        
    def remove(self, elemento):
        for tupla in self.pila:
            if tupla[0] == elemento:
                self.pila.remove(tupla)
        
    def elementos(self):
        return [x[0] for x in self.pila]
    
    def is_empty(self):
        return len(self.pila) == 0
    
    def __len__(self):
        return len(self.pila)

    def __str__(self):
        cadena = '['
        inicial = True
        for elemento, valor in self.pila:
            if inicial:
                cadena += '(' + str(elemento) + ',' + str(valor) + ')'
                inicial = False
            else:
                cadena += ', (' + str(elemento) + ',' + str(valor) + ')'
        return cadena + ']'
    
# valorNodo = namedtuple('valorNodo', ['recompensa', 'len_playout'])
class valorNodo :
    def __init__(self, recompensa:int, len_playout:int):
        self.recompensa = recompensa
        self.len_playout = len_playout
    def __str__(self) -> str:
        return f'{self.recompensa}/{self.len_playout}'

class ArbolBusquedaJuego :

    def __init__(self, state:any, juego:any, rollout_policy:any, sim_lim:int, beam_width:int, ucb_constant:float):
        self.juego = juego
        self.rollout_policy = rollout_policy
        self.sim_lim = sim_lim
        self.beam_width = beam_width
        self.raiz = NodoBusquedaJuego(state, None, None, valorNodo(0,0), ucb_constant)
        self.total = 0
        self.hojas = ListaPrioritaria()
        self.ucb_constant = ucb_constant
        self.dict_acciones = self.expandir_raiz()

    def get_argmax_acciones(self):
        valores = ListaPrioritaria()
        for accion in self.dict_acciones.keys():
            valor_nodo = self.dict_acciones[accion]
            valor = valor_nodo.recompensa / valor_nodo.len_playout if valor_nodo.len_playout != 0 else 0
            valores.push(accion, -valor) # Minus to invert the order
        str_accion = valores.pop()
        opt = [a for a in self.acciones if str(a) == str_accion]
        assert(len(opt) > 0), f'Oops! No hay acciones para escoger'
        return opt[0]

    def expandir(self, nodo):
        # Encontramos las acciones posibles
        acciones = self.juego.acciones(nodo.estado)
        if len(acciones) == 0:
            return None
        # Limitamos la cantidad de acciones al tamaño del haz
        if len(acciones) > self.beam_width:
            acciones = sample(acciones, self.beam_width)
        # Creamos los nodos correspondientes
        hijos = []
        for a in acciones:
            resultado_blancas = self.juego.resultado(nodo.estado, a)
            if self.juego.es_terminal(resultado_blancas):
                if self.juego.utilidad(resultado_blancas, 'blancas') > 0:
                    valor = 1
                else:
                    valor = 0
                self.backup_value(nodo, valor)
                continue
            accion_negras = self.rollout_policy(resultado_blancas)
            resultado_negras = self.juego.resultado(resultado_blancas, accion_negras)
            hijo = NodoBusquedaJuego(estado=resultado_negras,
                                     madre=nodo,
                                     accion=a,
                                     valor=valorNodo(0,0),
                                     ucb_constant=self.ucb_constant)
            hijos.append(hijo)
        if len(hijos) == 0:
            return None
        # Seleccionamos aleatoriamente una de las acciones
        seleccionado = choice(hijos)
        hijos.remove(seleccionado)
        # Actualizamos la frontera
        for hijo in hijos:
            self.hojas.push(hijo, hijo.ucb(self.total)) # Mejor valor es más pequeño en ListaPrioritaria
        # Retornamos el hijo seleccionado
        return seleccionado

    def expandir_raiz(self) -> dict:
        nodo = self.raiz
        # Encontramos las acciones posibles
        acciones = self.juego.acciones(nodo.estado)
        if len(acciones) == 0:
            return []
        # Limitamos la cantidad de acciones al tamaño del haz
        if len(acciones) > self.beam_width:
            acciones = sample(acciones, self.beam_width)
        self.acciones = acciones
        # Creamos los nodos correspondientes
        hijos = []
        dict_acciones = {str(a):valorNodo(0,0) for a in acciones}
        for a in acciones:
            resultado_blancas = self.juego.resultado(nodo.estado, a)
            if self.juego.es_terminal(resultado_blancas):
                dict_acciones[str(a)] = valorNodo(1,1)
                continue
            accion_negras = self.rollout_policy(resultado_blancas)
            resultado_negras = self.juego.resultado(resultado_blancas, accion_negras)
            hijo = NodoBusquedaJuego(resultado_negras, nodo, a, valorNodo(0,0), self.ucb_constant)
            hijos.append(hijo)
        # Actualizamos la frontera
        for hijo in hijos:
            self.hojas.push(hijo, hijo.ucb(self.total)) # Mejor valor es más pequeño en ListaPrioritaria
        return dict_acciones

    def seleccionar_ucb(self):
        return self.hojas.pop()
    
    def backup_value(self, nodo, resultado):
        self.total += 1
        # Retro propagamos el resultado
        accion, valor = nodo.retro_propagar(resultado)
        assert(str(accion) in [str(a) for a in self.dict_acciones.keys()]), f'{accion} not in {self.dict_acciones.keys()}'
        self.dict_acciones[str(accion)] = valor
        # Actualizamos las hojas con nuevos valores
        nuevas_hojas = ListaPrioritaria()
        for hoja in self.hojas.elementos():
            nuevas_hojas.push(hoja, hoja.ucb(self.total))
        self.hojas = nuevas_hojas


class NodoBusquedaJuego:

    def __init__ (self, estado:any, madre:any, accion:any, valor:valorNodo, ucb_constant:float):
        self.estado = estado
        self.madre = madre
        self.accion = accion
        self.valor = valor
        self.ucb_constant = ucb_constant
       
    def retro_propagar(self, resultado):
        self.valor.recompensa += resultado
        self.valor.len_playout += 1
        prof = self.profundidad()
        # print(f'{self.accion} a profundidad {prof} con madre {self.madre.accion}:\n\t  {self.valor.recompensa} / {self.valor.len_playout}')
        if prof > 1:
            return self.madre.retro_propagar(resultado)
        elif prof == 1:
            return self.accion, self.valor

    def ucb(self, total):
        numerador =  np.log(total + 1) if total > 0 else 1
        ucb_value = np.sqrt(numerador / (self.valor.len_playout + 1))
        return self.get_value() + self.ucb_constant * ucb_value

    def get_value(self):
        return self.valor.recompensa / self.valor.len_playout if self.valor.len_playout != 0 else 0

    def profundidad(self):
        if self.madre is None:
            return 0
        else:
            return 1 + self.madre.profundidad()
    
    def __str__(self):
        raiz = '--raiz--' if self.madre is None else ''
        cadena = "Estado:" + raiz + "\n" + str(self.estado) + "\n"
        cadena += "Profundidad: " + str(self.profundidad()) + "\n"
        if self.madre is not None:
            cadena += "Proviene de:\n" + str(self.madre.estado) + "\n"
            cadena += "Mediante acción: " + str(self.madre.estado.san(self.accion)) + "\n"
            cadena += "Valor: " + str(self.valor) + "\n"
            # cadena += "UBC: " + str(self.ucb()) + "\n"
        return cadena
